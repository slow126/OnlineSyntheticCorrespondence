/* 
Background Mandelbulb shader - scaled up version for use as background
Compatible with multi-object rendering system
*/
#version 430

// view
uniform vec3      iResolution;
uniform vec2      iViewAngleXY;
uniform float     iViewDistance   = 3.0;
uniform float     iFocalPlane     = 2.0;

// scene - enhanced lighting
uniform vec3      iMaterial       = vec3(1.0, 0.43, 0.25);
uniform vec2      iLightSource    = vec2(0.0);
uniform float     iAmbientLight   = 0.3;
uniform float     iDiffuseScale   = 0.6;
uniform float     iSpecularScale  = 0.45;
uniform float     iSpecularExp    = 10.0;

// rendering
uniform int       iAntialias      = 1;

// fractal
uniform vec4      iJuliaC;

// texture
uniform bool      iUseObjTexture  = false;
uniform sampler3D iObjTexture;
uniform bool      iUseBgTexture   = false;
uniform sampler2D iBgTexture;
uniform vec3      iObjectOffset;
uniform float     iMandelbulbP;

in vec2 v_texcoord;
in flat int v_object_id;

float POWER = 8;
float SCALE = 25.0;  // Scale factor to make mandelbulb much larger
float DEPTH_OFFSET = 15.0;  // Push mandelbulb back in depth (positive = further from camera)

#define ESCAPE_THRESHOLD 2.0
#define MAX_ITERATIONS 32
#define DEL 0.00005
#define BOUNDING_RADIUS_2 (100.0 * SCALE * SCALE)  // Much larger bounding radius

#define PI 3.14159265359

float intersection_count = 0.0;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 4) out float out_depth;
layout(location = 5) out int out_object_id;
layout(location = 6) out float out_distance_field;
layout(location = 7) out vec2 out_uv;

// Generate NO rotation for debugging - ensures identical appearance between views
mat3 getRandomRotation() {
    // Return identity matrix - no rotation for debugging
    return mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
}

// Enhanced distance estimation with AO
float mandelbulbDE(vec3 pos, out float AO) {
    // Scale it up (offset is handled in camera positioning)
    vec3 scaled_pos = pos / SCALE;
    vec3 z = scaled_pos;
    float dr = 1.0;
    float r = 0.0;
    AO = 1.0;
    
    // Remove bounding sphere logic to avoid smooth regions
    // Just do the mandelbulb calculation directly
    
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        // Darken as we go deeper (for AO effect)
        AO *= 0.725;
        
        r = length(z);
        if (r > ESCAPE_THRESHOLD) {
            // Point escaped - remap AO and return
            AO = min((AO + 0.075) * 4.1, 1.0);
            // Clamp the derivative to prevent division by very small numbers
            float clamped_dr = max(dr, 1e-6);
            return 0.5 * log(r) * r / clamped_dr * SCALE;
        }
        
        // Convert to polar coordinates
        float theta = acos(z.z/r);
        float phi = atan(z.y, z.x);
        dr = pow(r, POWER-1.0) * POWER * dr + 1.0;
        
        // Scale and rotate the point
        float zr = pow(r, POWER);
        theta = theta * POWER;
        phi = phi * POWER;
        
        // Convert back to cartesian coordinates
        z = zr * vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
        z += scaled_pos;
    }
    
    // Clamp the derivative to prevent division by very small numbers
    float clamped_dr = max(dr, 1e-6);
    return 0.5 * log(r) * r / clamped_dr * SCALE;
}

// Rotated version of mandelbulb distance estimation
float mandelbulbDE_rotated(vec3 pos, out float AO) {
    // Apply random rotation to the position
    mat3 rotation = getRandomRotation();
    vec3 rotated_pos = rotation * pos;
    
    return mandelbulbDE(rotated_pos, AO);
}

// Simple version without AO
float mandelbulbDE(vec3 p) {
    float ignore;
    return mandelbulbDE(p, ignore);
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(DEL, 0.0);
    vec3 n = vec3(
        mandelbulbDE(p + e.xyy) - mandelbulbDE(p - e.xyy),
        mandelbulbDE(p + e.yxy) - mandelbulbDE(p - e.yxy),
        mandelbulbDE(p + e.yyx) - mandelbulbDE(p - e.yyx)
    );
    
    // Clamp the normal length to prevent extreme values
    float len = length(n);
    if (len < 1e-6) {
        return vec3(0.0, 0.0, 1.0);  // Default normal if too small
    }
    return n / len;
}

// Calculate broad-scale normal for better shading
vec3 calcNormalBroad(vec3 p) {
    vec2 e = vec2(DEL * 50.0, 0.0);
    vec3 n = vec3(
        mandelbulbDE(p + e.xyy) - mandelbulbDE(p - e.xyy),
        mandelbulbDE(p + e.yxy) - mandelbulbDE(p - e.yxy),
        mandelbulbDE(p + e.yyx) - mandelbulbDE(p - e.yyx)
    );
    
    // Clamp the normal length to prevent extreme values
    float len = length(n);
    if (len < 1e-6) {
        return vec3(0.0, 0.0, 1.0);  // Default normal if too small
    }
    return n / len;
}

float intersect(inout vec3 rO, inout vec3 rD, float epsilon, out float AO) {
    float t = 0.0;
    float dist;
    AO = 1.0;
    
    // Increase iterations and use more conservative step size for better detail
    for(int i = 0; i < 400; i++) {
        dist = mandelbulbDE_rotated(rO, AO);
        t += dist * 0.8;  // More conservative step size for better detail
        rO += rD * dist * 0.8;
        
        if(dist < epsilon || t > 150.0)  // Increased max distance
            break;
    }
    return dist;
}

// Enhanced lighting with key/fill lights and AO
vec3 enhanced_lighting(vec3 light, vec3 eye, vec3 pt, vec3 N, vec3 N_broad, float AO) {
    vec3 baseColor = iMaterial;
    if (iUseObjTexture) {
        vec3 pp = pt * 0.5 + 0.5;
        baseColor = texture(iObjTexture, pp).rgb;
    }

    // Light direction (key light)
    vec3 L = normalize(light - pt);
    vec3 E = -eye;
    
    // Key and fill light colors
    const vec3 keyLightColor = vec3(1.0, 1.0, 1.0);
    const vec3 fillLightColor = vec3(0.0, 0.2, 0.7);
    
    // Mix between key and fill based on normal (Gooch-style wrap shading)
    float wrap = clamp(0.7 * dot(L, N) + 0.6, 0.0, 1.0);
    vec3 color = AO * mix(fillLightColor, keyLightColor, AO * wrap) * baseColor;
    
    // Highlight using broad normal
    vec3 highlight = AO * pow(max(dot(L, N_broad), 0.0), 5.0) * vec3(1.3, 1.2, 0.0);
    color += highlight;
    
    // gamma correction
    color = pow(color, vec3(1.0 / 2.2));
    
    return color;
}

vec4 render(vec3 ro, vec3 rd, vec3 light, float epsilon) {
    const vec4 backgroundColor = vec4(0.0);
    vec4 color = backgroundColor;
    vec3 N = vec3(0.0);
    vec3 N_broad = vec3(0.0);
    float AO = 1.0;

    float dist = intersect(ro, rd, epsilon, AO);
    out_distance_field = dist;

    if(dist < epsilon) {
        out_coord += ro;
        intersection_count += 1.0;
        
        // Calculate both fine and broad normals
        N = calcNormal(ro);
        N_broad = calcNormalBroad(ro);
        
        // Blend normals for better appearance
        N = normalize(N + N_broad + normalize(ro));
        
        color.xyz = enhanced_lighting(light, rd, ro, N, N_broad, AO);
        
        // Shadow calculation
        vec3 L = normalize(light - ro);
        vec3 p = ro + N * epsilon * 2.0;
        float dummy_ao;
        dist = intersect(p, L, epsilon, dummy_ao);
        if(dist < epsilon)
            color.xyz *= 0.8 + (dot(L, N) + 1.0) * 0.1;
        
        color.w = 1.0;
    }
    else {
        // Only render background if we have a valid mandelbulb hit
        // Sample at multiple distances along the ray to find the mandelbulb
        float min_dist = 1000.0;
        vec3 best_pos = vec3(0.0);
        float best_AO = 1.0;
        
        // Sample at multiple distances to find the mandelbulb with finer sampling
        for (float t = 3.0; t < 60.0; t += 2.0) {
            vec3 sample_pos = ro + rd * t;
            float sample_dist = mandelbulbDE_rotated(sample_pos, AO);
            if (sample_dist < min_dist) {
                min_dist = sample_dist;
                best_pos = sample_pos;
                best_AO = AO;
            }
        }
        
        // Only render if we found a reasonable mandelbulb distance
        if (min_dist < 10.0) {
            // Create background colors based on mandelbulb distance
            float dist_factor = smoothstep(0.0, 3.0, min_dist);
            vec3 bgColor1 = vec3(0.3, 0.1, 0.4) * max(best_AO, 0.1);
            vec3 bgColor2 = vec3(0.1, 0.3, 0.5) * max(best_AO, 0.1);
            
            color.xyz = mix(bgColor1, bgColor2, dist_factor);
            color.a = 1.0;
            
            // Set valid coordinates only for actual mandelbulb hits
            out_coord += best_pos;
            intersection_count += 1.0;
            out_depth = DEPTH_OFFSET + 10.0;
            out_object_id = v_object_id;
        } else {
            // No mandelbulb found - render nothing (black background)
            color = vec4(0.0, 0.0, 0.0, 0.0);
            // Don't set coordinates - they should remain 0,0,0 for invalid geometry
        }
    }

    out_normals += N;
    return color;
}

mat3 getCameraMatrix(vec3 camera, vec3 target, float roll) {
    vec3 u = vec3(sin(roll), cos(roll), 0.0);
    vec3 fw = normalize(target - camera);
    vec3 rt = normalize(cross(fw, u));
    vec3 up = normalize(cross(rt, fw));
    return mat3(rt, up, fw);
}

vec3 convertLocation(vec2 offsets, float distance) {
    float theta = PI * offsets.x;
    float phi = -PI * offsets.y;

    float cx = cos(phi), sx = sin(phi);
    float cy = cos(theta), sy = sin(theta);
    
    vec3 location = distance * vec3(sy * cx, -sx, cy * cx);
    return location;
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec3 res = iResolution;
    POWER = iMandelbulbP;

    vec3 lightSource = convertLocation(iLightSource.xy, 2 * iViewDistance);
    vec3 camera = convertLocation(iViewAngleXY.xy, iViewDistance);
    // Position mandelbulb to fill the background - closer and larger
    vec3 mandelbulb_offset = vec3(0.0, 0.0, DEPTH_OFFSET);
    camera += mandelbulb_offset;
    vec3 target = vec3(0.0) + mandelbulb_offset;
    float roll = 0.0;
    float focal = iFocalPlane;
    mat3 cameraMat = getCameraMatrix(camera, target, roll);
    out_coord = vec3(0.0);

    float mdim = max(res.x, res.y);
    int AA = max(1, iAntialias);
    vec4 color;
    float subpixel_size = AA;
    vec2 xy = (2.0 * fragCoord - res.xy) / mdim;
    
    for (int i = 0; i < AA; i++)
        for (int j = 0; j < AA; j++) {
            vec2 p = xy + vec2(float(i), float(j)) / (subpixel_size * mdim);
            vec3 rd = normalize(cameraMat * vec3(p, focal));
            color += render(camera, rd, lightSource, 5e-3);  // Finer epsilon for better detail
        }
    color /= float(AA * AA);
    out_coord /= max(1.0, intersection_count);
    
    float nlen = length(out_normals);
    out_normals = nlen > 0 ? normalize(out_normals / max(1.0, intersection_count)) : vec3(0.0);
    out_color = color;
    out_debug = vec3(v_texcoord, 1.0);
    
    // Output UV coordinates (normalized screen coordinates)
    out_uv = gl_FragCoord.xy / iResolution.xy;

    if (out_coord == vec3(0.0)) {
        out_depth = 200.0;
        out_object_id = -1;
        out_uv = vec2(0.0);  // Invalid UV for background
    }
    else {
        out_coord = out_coord - mandelbulb_offset;
        vec3 cameraToCoord = camera - out_coord;
        vec3 viewDir = normalize(camera - target);
        out_depth = dot(cameraToCoord, viewDir) + DEPTH_OFFSET;  // Add depth offset (further = larger depth value)
        out_object_id = v_object_id;
    }
}

