use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::prelude::*;
use bevy::tasks::{futures::check_ready, AsyncComputeTaskPool, Task};
use bevy::window::{CursorGrabMode, CursorOptions};
use fastnoise_lite::*;
use flagset::FlagSet;
use plumesplat::{ATTRIBUTE_MATERIAL_INDICES, ATTRIBUTE_MATERIAL_WEIGHTS, MaterialLayer};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};
use transvoxel::prelude::*;
use transvoxel::structs::transition_sides::TransitionSide;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CHUNK_SIZE: f32 = 100.0;
const SPHERE_RADIUS: f32 = 1000.0;
const GRID_EXTENT: i32 = 10; // chunks from -N..=N per axis

/// Subdivisions per LOD level (index 0 = highest detail).
const SUBDIVISIONS_MULT: usize = 5;
const SUBDIVISIONS: [usize; 6] = [64 * SUBDIVISIONS_MULT, 32 * SUBDIVISIONS_MULT, 16 * SUBDIVISIONS_MULT, 8 * SUBDIVISIONS_MULT, 4 * SUBDIVISIONS_MULT, 2 * SUBDIVISIONS_MULT];

/// Camera‑distance thresholds that decide the *desired* LOD.
const LOD_DISTANCES: [f32; 6] = [5.0, 20.9755146666666665, 100.654862, 260.404117333333332, 320.179916666666664, f32::MAX];
//const LOD_DISTANCES: [f32; 6] = [2.2340195555555553, 7.539815999999999, 17.872156444444443, 34.90655555555555, 60.31852799999999, f32::MAX];

/// Amplitude of the FBM noise displacement (world units).
const NOISE_AMPLITUDE: f32 = 70.0;
/// Spatial frequency for the FBM noise.
const NOISE_FREQUENCY: f32 = 0.0025;

/// Pre‑configured FBM noise generator (thread‑safe, lazily initialised).
static NOISE: LazyLock<FastNoiseLite> = LazyLock::new(|| {
    let mut n = FastNoiseLite::new();
    n.set_noise_type(Some(NoiseType::OpenSimplex2));
    n.set_fractal_type(Some(FractalType::Ridged));
    n.set_fractal_octaves(Some(5));
    n.set_frequency(Some(NOISE_FREQUENCY));
    n.set_fractal_lacunarity(Some(2.0));
    n.set_fractal_gain(Some(0.5));
    n
});

// ---------------------------------------------------------------------------
// Sphere SDF + FBM noise – the density field
// ---------------------------------------------------------------------------

fn sphere_density(x: f32, y: f32, z: f32) -> f32 {
    let sphere = 1.0 - (x * x + y * y + z * z).sqrt() / SPHERE_RADIUS;
    let fbm = NOISE.get_noise_3d(x, y, z); // −1..1
    sphere + fbm * (NOISE_AMPLITUDE / SPHERE_RADIUS)
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Marker for a chunk entity so we can find it back.
#[derive(Component)]
struct VoxelChunk {
    #[allow(dead_code)]
    coord: IVec3,
}

/// FPS‑style camera controller.
#[derive(Component)]
struct FpsCameraController {
    yaw: f32,
    pitch: f32,
    speed: f32,
    sensitivity: f32,
}

/// Async task for computing a chunk mesh on the thread pool.
#[derive(Component)]
struct ComputeChunkMesh {
    coord: IVec3,
    generation: u64,
    task: Task<Option<Mesh>>,
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Tracks every spawned chunk, the committed (displayed) state, and any
/// in‑flight batch of async mesh‑generation tasks.
#[derive(Resource, Default)]
struct ChunkManager {
    /// Entity per chunk coordinate.
    entities: HashMap<IVec3, Entity>,
    /// LOD state currently **displayed** (guaranteed crack‑free).
    committed_lod: HashMap<IVec3, usize>,
    committed_transitions: HashMap<IVec3, FlagSet<TransitionSide>>,
    /// Generation counter – bumped each time a new batch starts.
    generation: u64,
    /// Coords in the current in‑flight batch.
    batch_coords: HashSet<IVec3>,
    /// Mesh results collected so far for the current batch.
    batch_results: HashMap<IVec3, Option<Mesh>>,
    /// Full LOD / transition maps the current batch targets.
    batch_target_lod: HashMap<IVec3, usize>,
    batch_target_transitions: HashMap<IVec3, FlagSet<TransitionSide>>,
}

/// Stores the final LOD map after constraint propagation.
#[derive(Resource, Default)]
struct LodMap {
    lods: HashMap<IVec3, usize>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a chunk coordinate to the world‑space base (lowest corner).
fn chunk_base(coord: IVec3) -> [f32; 3] {
    [
        coord.x as f32 * CHUNK_SIZE,
        coord.y as f32 * CHUNK_SIZE,
        coord.z as f32 * CHUNK_SIZE,
    ]
}

/// Centre of the chunk in world space.
fn chunk_center(coord: IVec3) -> Vec3 {
    let b = chunk_base(coord);
    Vec3::new(
        b[0] + CHUNK_SIZE * 0.5,
        b[1] + CHUNK_SIZE * 0.5,
        b[2] + CHUNK_SIZE * 0.5,
    )
}

/// Desired LOD for a chunk given camera position (before constraint enforcement).
fn desired_lod(coord: IVec3, cam_pos: Vec3) -> usize {
    let dist = cam_pos.distance(chunk_center(coord));
    for (i, &threshold) in LOD_DISTANCES.iter().enumerate() {
        if dist < threshold {
            return i;
        }
    }
    SUBDIVISIONS.len() - 1
}

/// Enforce the constraint that adjacent chunks differ by at most 1 LOD level.
/// Iterates until stable.
fn enforce_lod_constraints(lods: &mut HashMap<IVec3, usize>) {
    let offsets: [IVec3; 6] = [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ];
    loop {
        let mut changed = false;
        let snapshot: Vec<(IVec3, usize)> = lods.iter().map(|(&k, &v)| (k, v)).collect();
        for (coord, lod) in &snapshot {
            for off in &offsets {
                let nb = *coord + *off;
                if let Some(&nb_lod) = lods.get(&nb) {
                    // If neighbour is more than 1 level finer, we must get finer too.
                    if *lod > nb_lod + 1 {
                        lods.insert(*coord, nb_lod + 1);
                        changed = true;
                    }
                    // And vice‑versa.
                    if nb_lod > *lod + 1 {
                        lods.insert(nb, *lod + 1);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// Compute the transition sides needed for a chunk.
fn compute_transition_sides(
    coord: IVec3,
    lods: &HashMap<IVec3, usize>,
) -> FlagSet<TransitionSide> {
    let my_lod = match lods.get(&coord) {
        Some(&l) => l,
        None => return TransitionSide::none(),
    };
    let mut sides = TransitionSide::none();

    let check = |dir: IVec3, side: TransitionSide, sides: &mut FlagSet<TransitionSide>| {
        if let Some(&nb_lod) = lods.get(&(coord + dir)) {
            // Neighbour has *lower* LOD index → higher resolution → we are the
            // low‑res block and need a transition face toward it.
            if nb_lod < my_lod {
                *sides |= side;
            }
        }
    };

    check(IVec3::X, TransitionSide::HighX, &mut sides);
    check(IVec3::NEG_X, TransitionSide::LowX, &mut sides);
    check(IVec3::Y, TransitionSide::HighY, &mut sides);
    check(IVec3::NEG_Y, TransitionSide::LowY, &mut sides);
    check(IVec3::Z, TransitionSide::HighZ, &mut sides);
    check(IVec3::NEG_Z, TransitionSide::LowZ, &mut sides);

    sides
}

/// Generate a Bevy `Mesh` for a chunk via the transvoxel crate.
/// Returns `None` when the chunk contains no surface geometry.
fn generate_chunk_mesh(coord: IVec3, lod: usize, transitions: FlagSet<TransitionSide>) -> Option<Mesh> {
    let base = chunk_base(coord);
    let subdivisions = SUBDIVISIONS[lod];
    let block = Block::new(base, CHUNK_SIZE, subdivisions);
    let threshold = 0.0_f32;

    let builder = GenericMeshBuilder::new();
    let builder = extract_from_field(
        &sphere_density,
        FieldCaching::CacheNothing,
        block,
        transitions,
        threshold,
        builder,
    );
    let tv_mesh = builder.build();

    // Convert to Bevy mesh.
    let num_verts = tv_mesh.positions.len() / 3;
    if num_verts == 0 {
        return None;
    }
    let positions: Vec<[f32; 3]> = tv_mesh
        .positions
        .chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let normals: Vec<[f32; 3]> = tv_mesh
        .normals
        .chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let indices: Vec<u32> = tv_mesh.triangle_indices.iter().map(|&i| i as u32).collect();

    let mut material_indices: Vec<u32> = Vec::new();
    let mut material_weights: Vec<u32> = Vec::new();
     let mut uvs: Vec<[f32; 2]> = Vec::new();

    let dim = 1000.0 * 1.0;

    positions.iter().zip(normals.iter()).for_each(|(p, n)| {
        let vec3_p: Vec3 = Vec3::new(p[0], p[1], p[2]);
        let vec3_n: Vec3 = Vec3::new(n[0], n[1], n[2]);

        let height = (vec3_p.length() - SPHERE_RADIUS).abs(); // Y-axis as height
        let slope = vec3_p.normalize().dot(vec3_n).acos()*0.2; // Slope based on normal
        let blend = compute_material_blend_with_slope(height, 40.0, slope);
        // material_indices.push(blend.indices[0] as u32);
        // material_indices.push(blend.indices[1] as u32);
        // material_indices.push(blend.indices[2] as u32);
        // material_indices.push(blend.indices[3] as u32);

        // // Pack weights into 8 bits each (0-255)
        // let w0 = (blend.weights[0] as f32 * 255.0).round() as u32;
        // let w1 = (blend.weights[1] as f32 * 255.0).round() as u32;
        // let w2 = (blend.weights[2] as f32 * 255.0).round() as u32;
        // let w3 = (blend.weights[3] as f32 * 255.0).round() as u32;
        // material_weights.push((w0 << 24) | (w1 << 16) | (w2 << 8) | w3);

        uvs.push([
                p[0] as f32 / (dim - 1.0) as f32,
                p[1] as f32 / (dim - 1.0) as f32,
            ]);

        material_indices.push(blend.packed_indices());
        material_weights.push(blend.packed_weights());
    });

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);

    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(ATTRIBUTE_MATERIAL_INDICES, material_indices);
    mesh.insert_attribute(ATTRIBUTE_MATERIAL_WEIGHTS, material_weights);


    mesh.insert_indices(Indices::U32(indices));
    Some(mesh)
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Startup: spawn camera, light, and initial chunks.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    mut chunk_mgr: ResMut<ChunkManager>,
    mut lod_map: ResMut<LodMap>,
) {
    // Camera
    let cam_pos = Vec3::new(0.0, 0.0, 1450.0);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(cam_pos).looking_at(Vec3::ZERO, Vec3::Y),
        FpsCameraController {
            yaw: -90.0_f32.to_radians(), // looking toward -Z initially
            pitch: 0.0,
            speed: 10.0,
            sensitivity: 0.002,
        },
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
    ));
    // commands.spawn((
    //     PointLight {
    //         intensity: 200_000.0,
    //         range: 200.0,
    //         ..default()
    //     },
    //     Transform::from_xyz(30.0, 30.0, 30.0),
    // ));

    // Material shared by all chunks
    let a_material = StandardMaterial {
        base_color: Color::srgb(0.4, 0.7, 0.3),
        perceptual_roughness: 0.85,
        metallic: 0.0,
        double_sided: true,
        cull_mode: None,
        ..default()
    };

    // Build initial LOD map
    let mut lods = HashMap::new();
    for x in -GRID_EXTENT..=GRID_EXTENT {
        for y in -GRID_EXTENT..=GRID_EXTENT {
            for z in -GRID_EXTENT..=GRID_EXTENT {
                let c = IVec3::new(x, y, z);
                lods.insert(c, desired_lod(c, cam_pos));
            }
        }
    }
    enforce_lod_constraints(&mut lods);

    // Compute transitions for the initial LOD map.
    let mut all_transitions: HashMap<IVec3, FlagSet<TransitionSide>> = HashMap::new();
    for &coord in lods.keys() {
        all_transitions.insert(coord, compute_transition_sides(coord, &lods));
    }

    // Define individual material layers using the builder API
    // Each layer loads its own textures from the raw folder
    let grass = MaterialLayer::new(asset_server.load("textures/raw/Grass004_1K-PNG_Color.png"))
        .with_normal(asset_server.load("textures/raw/Grass004_1K-PNG_NormalGL.png"));
    let grass = Arc::new(grass);
    let dirt = MaterialLayer::new(asset_server.load("textures/raw/Ground054_1K-PNG_Color.png"))
        .with_normal(asset_server.load("textures/raw/Ground054_1K-PNG_NormalGL.png"));
    let dirt = Arc::new(dirt);

    let rock = MaterialLayer::new(asset_server.load("textures/raw/Rock051_1K-PNG_Color.png"))
        .with_normal(asset_server.load("textures/raw/Rock051_1K-PNG_NormalGL.png"));
    let rock = Arc::new(rock);

    // Snow only has a diffuse texture available
    let snow = MaterialLayer::new(asset_server.load("textures/raw/snow_diff.png"));
    let snow = Arc::new(snow);
    // Build the pending material - it will automatically convert to a
    // full PlumeSplatMaterial when all textures finish loading
    let pending_material = plumesplat::SplatMaterialBuilder::new()
    .with_base_material(a_material)
        .add_layer(grass)
        .add_layer(dirt)
        .add_layer(rock)
        .add_layer(snow)
        .with_uv_scale(0.9)
        .with_triplanar_sharpness(1.0)
        .with_blend_offset(0.0)
        .with_blend_exponent(2.0)
        .build();

    // Spawn chunk entities with empty placeholder meshes.
    for &coord in lods.keys() {
        let handle = meshes.add(Mesh::new(PrimitiveTopology::TriangleList, default()));
        let entity = commands
            .spawn((
                Mesh3d(handle),
                pending_material.clone(),//MeshMaterial3d(material.clone()),
                //MeshMaterial3d(a_material.clone()),
                Transform::default(),
                Visibility::Hidden,
                VoxelChunk { coord },
            ))
            .id();
        chunk_mgr.entities.insert(coord, entity);
    }

    // Dispatch the initial batch – every chunk is dirty (committed is empty).
    chunk_mgr.generation = 1;
    let batch_gen = chunk_mgr.generation;
    let pool = AsyncComputeTaskPool::get();
    let batch: HashSet<IVec3> = lods.keys().copied().collect();
    for &coord in &batch {
        let lod = lods[&coord];
        let trans = all_transitions[&coord];
        let task = pool.spawn(async move { generate_chunk_mesh(coord, lod, trans) });
        commands.spawn(ComputeChunkMesh { coord, generation: batch_gen, task });
    }
    chunk_mgr.batch_coords = batch;
    chunk_mgr.batch_target_lod = lods.clone();
    chunk_mgr.batch_target_transitions = all_transitions;

    lod_map.lods = lods;
}

/// FPS camera movement (WASD + Q/E for up/down, mouse look).
fn fps_camera_system(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    accumulated_mouse: Res<AccumulatedMouseMotion>,
    mut query: Query<(&mut Transform, &mut FpsCameraController)>,
    mut cursor_query: Query<&mut CursorOptions>,
) {
    let Ok((mut transform, mut ctrl)) = query.single_mut() else {
        return;
    };

    // --- Mouse look ---
    let mouse_delta = -accumulated_mouse.delta;
    ctrl.yaw -= mouse_delta.x * ctrl.sensitivity;
    ctrl.pitch -= mouse_delta.y * ctrl.sensitivity;
    ctrl.pitch = ctrl
        .pitch
        .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());

    let (sin_p, cos_p) = ctrl.pitch.sin_cos();
    let (sin_y, cos_y) = ctrl.yaw.sin_cos();
    let forward = Vec3::new(cos_p * cos_y, sin_p, cos_p * sin_y).normalize();
    let right = Vec3::Y.cross(forward).normalize();
    let up = Vec3::Y;

    transform.rotation = Quat::from_rotation_arc(Vec3::NEG_Z, forward);

    // --- Keyboard movement ---
    let dt = time.delta_secs();
    let speed = ctrl.speed * if keys.pressed(KeyCode::ShiftLeft) { 3.0 } else { 1.0 };

    if keys.pressed(KeyCode::KeyW) {
        transform.translation += forward * speed * dt;
    }
    if keys.pressed(KeyCode::KeyS) {
        transform.translation -= forward * speed * dt;
    }
    if keys.pressed(KeyCode::KeyA) {
        transform.translation += right * speed * dt;
    }
    if keys.pressed(KeyCode::KeyD) {
        transform.translation -= right * speed * dt;
    }
    if keys.pressed(KeyCode::KeyQ) {
        transform.translation -= up * speed * dt;
    }
    if keys.pressed(KeyCode::KeyE) {
        transform.translation += up * speed * dt;
    }

    // Grab the cursor so the mouse look works seamlessly.
    if let Ok(mut cursor_opts) = cursor_query.single_mut() {
        cursor_opts.grab_mode = CursorGrabMode::Locked;
        cursor_opts.visible = false;
    }
}

/// Every frame, recompute LODs.  A new batch is dispatched only when no batch
/// is currently in flight, so the displayed state is always a consistent
/// LOD‑map snapshot (no cracks).
fn update_lod_system(
    mut commands: Commands,
    cam_query: Query<&Transform, With<FpsCameraController>>,
    mut chunk_mgr: ResMut<ChunkManager>,
    mut lod_map: ResMut<LodMap>,
) {
    // Don’t start a new batch while one is still in flight.
    if !chunk_mgr.batch_coords.is_empty() {
        return;
    }

    let Ok(cam_tf) = cam_query.single() else {
        return;
    };
    let cam_pos = cam_tf.translation;

    // 1. Compute desired LODs.
    let mut lods: HashMap<IVec3, usize> = HashMap::new();
    for x in -GRID_EXTENT..=GRID_EXTENT {
        for y in -GRID_EXTENT..=GRID_EXTENT {
            for z in -GRID_EXTENT..=GRID_EXTENT {
                let c = IVec3::new(x, y, z);
                lods.insert(c, desired_lod(c, cam_pos));
            }
        }
    }

    // 2. Enforce constraints.
    enforce_lod_constraints(&mut lods);

    // 3. Compute transitions for the whole map.
    let mut transitions: HashMap<IVec3, FlagSet<TransitionSide>> = HashMap::new();
    for &coord in lods.keys() {
        transitions.insert(coord, compute_transition_sides(coord, &lods));
    }

    // 4. Find chunks that differ from the committed (displayed) state.
    let mut dirty: HashSet<IVec3> = HashSet::new();
    for (&coord, &lod) in &lods {
        let c_lod = chunk_mgr.committed_lod.get(&coord).copied();
        let c_trans = chunk_mgr.committed_transitions.get(&coord).copied();
        if c_lod != Some(lod) || c_trans != Some(transitions[&coord]) {
            dirty.insert(coord);
        }
    }

    if dirty.is_empty() {
        lod_map.lods = lods;
        return;
    }

    // 5. Dispatch a new batch.
    chunk_mgr.generation += 1;
    let batch_gen = chunk_mgr.generation;
    let pool = AsyncComputeTaskPool::get();
    for &coord in &dirty {
        let lod = lods[&coord];
        let trans = transitions[&coord];
        let task = pool.spawn(async move { generate_chunk_mesh(coord, lod, trans) });
        commands.spawn(ComputeChunkMesh { coord, generation: batch_gen, task });
    }
    chunk_mgr.batch_coords = dirty;
    chunk_mgr.batch_results.clear();
    chunk_mgr.batch_target_lod = lods.clone();
    chunk_mgr.batch_target_transitions = transitions;

    lod_map.lods = lods;
}

/// Poll completed tasks, buffer results, and apply the entire batch atomically
/// so that all visible chunks are always consistent (crack‑free).
fn handle_mesh_tasks(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut ComputeChunkMesh)>,
    mut chunk_mgr: ResMut<ChunkManager>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunk_query: Query<(&Mesh3d, &mut Visibility), With<VoxelChunk>>,
) {
    let current_gen = chunk_mgr.generation;

    // 1. Collect completed results.
    for (task_entity, mut task) in &mut tasks {
        if let Some(maybe_mesh) = check_ready(&mut task.task) {
            if task.generation == current_gen {
                chunk_mgr.batch_results.insert(task.coord, maybe_mesh);
            }
            // Stale or current – always despawn the task entity.
            commands.entity(task_entity).despawn();
        }
    }

    // 2. If the entire batch is complete, apply ALL results at once.
    if chunk_mgr.batch_coords.is_empty() {
        return;
    }
    let all_done = chunk_mgr
        .batch_coords
        .iter()
        .all(|c| chunk_mgr.batch_results.contains_key(c));
    if !all_done {
        return;
    }

    let batch = std::mem::take(&mut chunk_mgr.batch_coords);
    let mut results = std::mem::take(&mut chunk_mgr.batch_results);

    for coord in &batch {
        if let Some(maybe_mesh) = results.remove(coord) {
            if let Some(&entity) = chunk_mgr.entities.get(coord) {
                if let Ok((mesh_handle, mut vis)) = chunk_query.get_mut(entity) {
                    match maybe_mesh {
                        Some(new_mesh) => {
                            let _ = meshes.insert(&mesh_handle.0, new_mesh);
                            *vis = Visibility::Visible;
                        }
                        None => {
                            *vis = Visibility::Hidden;
                        }
                    }
                }
            }
        }
    }

    // Commit: the displayed state now matches the batch target.
    chunk_mgr.committed_lod = std::mem::take(&mut chunk_mgr.batch_target_lod);
    chunk_mgr.committed_transitions = std::mem::take(&mut chunk_mgr.batch_target_transitions);
}

/// Toggle global wireframe on/off with F1.
fn toggle_wireframe(keys: Res<ButtonInput<KeyCode>>, mut config: ResMut<WireframeConfig>) {
    if keys.just_pressed(KeyCode::F1) {
        config.global = !config.global;
    }
}

/// Compute material blend based on height alone.
fn compute_material_blend(height: f32, max_height: f32) -> plumesplat::MaterialVertex {
    let normalized_height = height / max_height;

    // Define height bands for each material
    // 0: Grass (0.0 - 0.3)
    // 1: Dirt (0.2 - 0.5)
    // 2: Rock (0.4 - 0.8)
    // 3: Snow (0.7 - 1.0)

    let grass = smooth_band(normalized_height, 0.0, 0.35, 0.1);
    let dirt = smooth_band(normalized_height, 0.25, 0.55, 0.1);
    let rock = smooth_band(normalized_height, 0.45, 0.85, 0.15);
    let snow = smooth_band(normalized_height, 0.7, 1.0, 0.1);

    plumesplat::MaterialVertex::blend4([0, 1, 2, 3], [grass, dirt, rock, snow])
}

/// Compute material blend based on height and slope.
fn compute_material_blend_with_slope(height: f32, max_height: f32, slope: f32) -> plumesplat::MaterialVertex {
    let normalized_height = height / max_height;

    // Base height-based weights with wider fade zones for smoother blending
    let grass = smooth_band(normalized_height, 0.0, 0.4, 0.2);
    let dirt = smooth_band(normalized_height, 0.2, 0.6, 0.2);
    let rock = smooth_band(normalized_height, 0.4, 0.9, 0.25);
    let snow = smooth_band(normalized_height, 0.65, 1.0, 0.2);

    // Steep slopes get more rock
    let slope_factor = (slope * 3.0).clamp(0.0, 1.0);
    let rock = rock + slope_factor * 0.5;
    let grass = grass * (1.0 - slope_factor * 0.7);
    let snow = snow * (1.0 - slope_factor * 0.5);

    plumesplat::MaterialVertex::blend4([0, 1, 2, 3], [grass, dirt, rock, snow])
}

/// Smooth band function for height-based material blending.
fn smooth_band(value: f32, start: f32, end: f32, fade: f32) -> f32 {
    let fade_in = smoothstep(start - fade, start + fade, value);
    let fade_out = 1.0 - smoothstep(end - fade, end + fade, value);
    fade_in * fade_out
}

/// Hermite smoothstep interpolation.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyBlocks – Transvoxel LOD Sphere".into(),
                //resolution: bevy::window::WindowResolution::new(640, 360),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(WireframePlugin::default())
        .add_plugins(plumesplat::PlumeSplatPlugin)
        .init_resource::<ChunkManager>()
        .init_resource::<LodMap>()
        .add_systems(Startup, setup)
        .add_systems(Update, (fps_camera_system, update_lod_system, handle_mesh_tasks, toggle_wireframe))
        .run();
}
