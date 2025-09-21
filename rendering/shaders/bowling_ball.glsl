#version 430
#define HW_PERFORMANCE 1
// view
uniform vec3      iResolution;
uniform vec2      iViewAngleXY;
uniform float     iViewDistance   = 3.0;
uniform float     iFocalPlane     = 2.0;
// scene
uniform vec3      iMaterial       = vec3(1.0, 0.43, 0.25);
uniform vec2      iLightSource    = vec2(0.0);
uniform float     iAmbientLight   = 0.3;
uniform float     iDiffuseScale   = 0.6;
uniform float     iSpecularScale  = 0.45;
uniform float     iSpecularExp    = 10.0;
// rendering
uniform int       iAntialias      = 1;
// fractal
uniform float      iHashSeed       = 43758.5453123;
// texture
uniform bool      iUseObjTexture  = false;
uniform sampler3D iObjTexture;
uniform bool      iUseBgTexture   = false;
uniform sampler2D iBgTexture;
uniform vec3      iObjectOffset;

in vec2 v_texcoord;
in flat int v_object_id;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 4) out float out_depth;
layout(location = 5) out int out_object_id;

vec3 camera_perspective(vec3 lookfrom, vec3 lookat, float tilt, float vfov, vec2 uv);
float sphereSDF(vec3 p, vec3 center, float radius);
vec2 scene_map(vec3 p);
vec2 ray_march(vec3 ro, vec3 rd, out vec3 hitPoint);
vec3 getGaussianNoiseColor(vec3 p);

vec3 fade(vec3 t);
float perlinNoise(vec3 p);
vec3 getPerlinNoiseColor(vec3 p, float scale);

vec3 get_obj_color_from_volume(vec3 p);
vec3 generate_obj_color(vec3 p);
void mainImage(out vec4 frag_col, in vec2 frag_coord);


float hash1(float n) {
    return fract(sin(n) * iHashSeed);
}

float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * iHashSeed);
}

float hash3(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * iHashSeed);
}

vec3 random3(vec3 p) {
    // 3D pseaudo-random for noise
    return vec3(hash3(p), hash3(p + vec3(0.0, 1.0, 0.0)), hash3(p + vec3(0.0, 0.0, 1.0)));
}

vec3 camera_perspective(vec3 lookfrom, vec3 lookat, float tilt, float vfov, vec2 uv) {
  vec2 sc = vec2(sin(tilt), cos(tilt));
  vec3 vup = vec3(sc.x, sc.y, 0.0);
  vec3 w = normalize((lookat - lookfrom));
  vec3 u = normalize(cross(w, vup));
  vec3 v = cross(u, w);
  float wf = (1.0 / tan((vfov * 0.00872664626)));
  return normalize((((uv.x * u) + (uv.y * v)) + (wf * w)));
}

float sphereSDF(vec3 p, vec3 center, float radius) {
    return length(p - center) - radius;
}

float BlobSDF(vec3 p, vec3 center,  float r) {
    // Base sphere distance
    float baseDistance = length(p - center) - r;

    // Perturbation using a noise-like function
    vec3 m1 = vec3(1.2, 0.3, .1);
    vec3 f1 = vec3(0.7, 0.3, 2.3);
    vec3 a1 = vec3(1.3, 2, 4.0);
    float noise1 = sin((p.x + m1.x) * f1.x) * a1.x * sin((p.y + m1.y) * f1.y) * a1.y * sin((p.z + m1.z) * f1.z) * a1.z;

    float noise2 = sin(p.x * 4.0) * sin(p.y * 7.0) * sin(p.z * 5.0);
    float a3 = 0.6;
    float noise3 = sin(p.x * 15.0) * sin(p.y * 12.0) * sin(p.z * 9.0) * a3;

    // Add the perturbation to the sphere distance
    return baseDistance + 0.3 * noise1 + 0.2 * noise2 + 0.1 * noise3;
}

vec2 scene_map(vec3 p) {
  // Object
  float materialID = 0.0;
  vec3 objCenter = vec3(0.0, 0.5, 0.0); // Example position
  float objRadius = 0.2;
  float objDist = sphereSDF(p, objCenter, objRadius);
//   float objDist = BlobSDF(p, vec3(0.0, 0.5, 0.0), .01);

  return vec2(objDist * 0.5, materialID);
}

vec2 ray_march(vec3 ro, vec3 rd, out vec3 hitPoint) {
  float t = 0.0;
  float materialID = 0.0;
  for (int i = 1; i <= 256; i = i + 1) {
    vec3 p = ro + t * rd;
    vec2 mapResult = scene_map(p);
    float d = mapResult.x;
    materialID = mapResult.y;
    if ((d < (0.003 * t)) || (t >= 25.0)) {
      hitPoint = p; // Save the final point
      break;
    }
    t = t + d;
  }
  return vec2(t, materialID);
}

vec3 getUniformNoiseColor(vec3 p, float scale) {
    vec3 scaled_p = p / 50.;
    vec3 noise = random3((p));
    noise /= 1.0;
    return noise;
}

vec3 fade(vec3 t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float perlinNoise(vec3 p) {
    vec3 pi = floor(p);  // Grid cell coordinates
    vec3 pf = fract(p);  // Local coordinates within cell
    
    // Fade function to smooth interpolation
    vec3 f = fade(pf);
    
    // Hashes for gradient selection at cube corners
    vec3 g000 = random3(pi);
    vec3 g001 = random3(pi + vec3(0.0, 0.0, 1.0));
    vec3 g010 = random3(pi + vec3(0.0, 1.0, 0.0));
    vec3 g011 = random3(pi + vec3(0.0, 1.0, 1.0));
    vec3 g100 = random3(pi + vec3(1.0, 0.0, 0.0));
    vec3 g101 = random3(pi + vec3(1.0, 0.0, 1.0));
    vec3 g110 = random3(pi + vec3(1.0, 1.0, 0.0));
    vec3 g111 = random3(pi + vec3(1.0, 1.0, 1.0));

    // Compute dot product between gradient and local position
    float v000 = dot(g000, pf);
    float v001 = dot(g001, pf - vec3(0.0, 0.0, 1.0));
    float v010 = dot(g010, pf - vec3(0.0, 1.0, 0.0));
    float v011 = dot(g011, pf - vec3(0.0, 1.0, 1.0));
    float v100 = dot(g100, pf - vec3(1.0, 0.0, 0.0));
    float v101 = dot(g101, pf - vec3(1.0, 0.0, 1.0));
    float v110 = dot(g110, pf - vec3(1.0, 1.0, 0.0));
    float v111 = dot(g111, pf - vec3(1.0, 1.0, 1.0));

    // Trilinear interpolation using fade factors
    float x1 = mix(v000, v100, f.x);
    float x2 = mix(v010, v110, f.x);
    float x3 = mix(v001, v101, f.x);
    float x4 = mix(v011, v111, f.x);

    float y1 = mix(x1, x2, f.y);
    float y2 = mix(x3, x4, f.y);

    return mix(y1, y2, f.z);
}

// Use Perlin noise for color generation
vec3 getPerlinNoiseColor(vec3 p, float scale) {
    float nr = perlinNoise(p * scale);              // Red channel noise
    float ng = perlinNoise(p * scale + vec3(13.7)); // Green channel noise (offset)
    float nb = perlinNoise(p * scale + vec3(31.4)); // Blue channel noise (offset)
    float brighten = 6.;
    return vec3(nr*brighten, ng*brighten, nb*brighten);
}

vec3 getLatStripeTexture(vec3 p, float frequency, float transitionHardness, float offset) {
    float stripePattern = sin((p.y) * frequency * 3.14159+ offset); // Stripes along X-axis
    float stripeValue = smoothstep(-1.0 + transitionHardness, 1.0 - transitionHardness, stripePattern);
    
    return vec3(stripeValue); // Black & White stripes
}

vec3 getLonStripeTexture(vec3 p, float frequency, float transitionHardness, float offset) {
    float stripePattern = sin((p.x) * frequency * 3.14159 + offset); // Stripes along X-axis
    float stripeValue = smoothstep(-1.0 + transitionHardness, 1.0 - transitionHardness, stripePattern);
    
    return vec3(stripeValue); // Black & White stripes
}

float worleyNoise(vec3 p) {
    vec3 pi = floor(p); // Grid cell
    vec3 pf = fract(p); // Position within cell
    
    float minDist = 1.0; // Store the minimum distance

    // Search in a 3x3x3 grid around the current cell
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                vec3 neighbor = vec3(float(x), float(y), float(z));
                vec3 featurePoint = random3(pi + neighbor); // Random point in cell
                float dist = length(featurePoint + neighbor - pf); // Euclidean distance
                minDist = min(minDist, dist); // Keep the closest distance
            }
        }
    }
    
    return minDist; // Output closest distance as noise
}

vec3 getWorleyNoiseColor(vec3 p, vec3 scale) {
    float nr = worleyNoise(p * scale);
    float ng = worleyNoise(p * scale + vec3(12.3));
    float nb = worleyNoise(p * scale + vec3(45.6));

    return vec3(nr, ng, nb) ;
}


vec3 get_obj_color_from_volume(vec3 p) {
    // Dual-freq perlin
    vec3 low_freq_noise = getPerlinNoiseColor(p, 2.);
    vec3 high_freq_noise = getPerlinNoiseColor(p, 10.);
    // return mix(low_freq_noise, high_freq_noise, 0.5);

    // Gaussian
    // return getGaussianNoiseColor(p, 10.);

    // Lat stripe texture
    // return getLatStripeTexture(p, 20.);

    // Lon Stripe texture
    // return getLonStripeTexture(p, 20.);

    // Texture Mix
    vec3 stripes = mix(getLatStripeTexture(p, 15., .2, 0.7)*vec3(0., 1., 0.), getLonStripeTexture(p, 2., .9, 3.5)*vec3(1., 0., 0.), .5);
    vec3 perlin = mix(low_freq_noise, high_freq_noise, 1.);
    vec3 noise = getUniformNoiseColor(p, 10.);
    vec3 med_f_worley = getWorleyNoiseColor(p, vec3(200.))*2.;
    vec3 high_f_worley = getWorleyNoiseColor(p, vec3(600.))*2.;
    vec3 worley_noise_mix = mix(med_f_worley, high_f_worley, 0.5);
    vec3 worley = getWorleyNoiseColor(p, vec3(20., 1., 15.))*2.;
    vec3 base_color = vec3(1., 0., .5);
    vec3 colored_worley = mix(worley, base_color, 0.4);
    vec3 greyed_worley = vec3(worley.x, worley.x, worley.x);
    vec3 noise_layered = mix(worley_noise_mix*vec3(.8, .4, .4), greyed_worley*vec3(.9, .6, 0.4), 0.7);
    return mix(noise_layered, stripes, 0.)*1.2;
}

vec3 generate_obj_color(vec3 p) {
    // Use x, y, z coordinates to generate the color noise
    return get_obj_color_from_volume(p);
}

void main() {
    vec4 frag_col;
    vec2 frag_coord = gl_FragCoord.xy;
    vec2 res = vec2(iResolution.x, iResolution.y);
    vec2 uv = (frag_coord / res);
    vec2 coord = ((2.0 * (frag_coord - (res * 0.5))) / res.y);

// camera as function of mouse position
    //float z = (iMouse.y * 0.01);
    vec3 origin_tr = vec3(0., 0., 0.);
    float z = 0.;
    float orbit_x = 0.;
    // float orbit_x = (iMouse.x * 0.02);
    vec2 sc = vec2(.5);
    // vec3 lookat = vec3((sc.x * 0.5), -0.7, z);
    vec3 lookat = origin_tr - vec3(0., -.45, 0.);
    // vec3 ro = vec3(((-sc.x) * 0.5), 0.2, (z - 2.0 + orbit_x));
    // camera orbit the origin as a function of the mouse x position
    float pseudo_dist = .8; // Camera distance
    vec3 ro = vec3(pseudo_dist * cos(orbit_x), .6, pseudo_dist * sin(orbit_x));
    vec3 rd = camera_perspective(ro, lookat, 0., 45.0, coord);
    

  // Ray marching
  vec3 hitPoint;
  vec2 march_result = ray_march(ro, rd, hitPoint);
  float t = march_result.x;
  float materialID = march_result.y;
  vec3 p = (ro + (rd * t));
  vec3 col = vec3(0.0);
  vec3 normal = vec3(0.);

  if (t < 25.0) {
    // Compute normal using gradient approximation
    float eps = 0.01;
    vec3 grad = vec3(
        scene_map(p + vec3(eps, 0.0, 0.0)).x - scene_map(p - vec3(eps, 0.0, 0.0)).x,
        scene_map(p + vec3(0.0, eps, 0.0)).x - scene_map(p - vec3(0.0, eps, 0.0)).x,
        scene_map(p + vec3(0.0, 0.0, eps)).x - scene_map(p - vec3(0.0, 0.0, eps)).x
    );
    normal = normalize(grad);

    // Phong lighting
    vec3 light_dir = normalize(vec3(0, 20, -100)); // Directional light
    vec3 view_dir = normalize(ro - p);               // View direction
    vec3 reflect_dir = reflect(-light_dir, normal);  // Reflection direction

    // Light colors
    vec3 ambient = vec3(0.3, 0.3, 0.3);
    vec3 diffuse = max(dot(light_dir, normal), 0.0) * vec3(0.5, 0.5, 0.5);
    vec3 specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * vec3(0.8, 0.8, 0.8);

    // Combine components
    col = ambient + diffuse + specular;

    // Color the object
    vec3 obj_col = generate_obj_color(p);
    

    if (materialID < 0.5) {
        // Object color
        col *= obj_col;
    } else {
        // background color
    }

    out_object_id = v_object_id;
    out_depth = t; // Double check if this is accurate. If not, set to cam pos - coord
    
  }
  else {
    // Sky color
    col = vec3(0.8, 0.9, 1.);
    out_object_id = 0;
    out_depth = 100;
  }

  // Output final color
  frag_col = vec4(col, 1.0);

  out_color = frag_col;
  out_coord = hitPoint; //TODO: Fix

  out_normals = normal;
  out_debug = vec3(0.0);
  
  
}