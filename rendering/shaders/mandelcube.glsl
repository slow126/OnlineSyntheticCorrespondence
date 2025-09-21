/* 
Mandelbox distance estimation shader
*/
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
// texture
uniform bool      iUseObjTexture  = false;
uniform sampler3D iObjTexture;
uniform bool      iUseBgTexture   = false;
uniform sampler2D iBgTexture;
uniform vec3      iObjectOffset;

in vec2 v_texcoord;
in flat int v_object_id;

#define ESCAPE_THRESHOLD 2.0
#define MAX_ITERATIONS 100
#define DEL 0.0001
#define BOUNDING_RADIUS_2 5

#define PI 3.14159265359

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 4) out float out_depth;
layout(location = 5) out int out_object_id;
layout(location = 6) out float out_distance_field;

float boxFold(float v, float fold) {
    if (v > fold) v = 2.0 * fold - v;
    else if (v < -fold) v = -2.0 * fold - v;
    return v;
}

vec3 boxFold(vec3 p, float fold) {
    p.x = boxFold(p.x, fold);
    p.y = boxFold(p.y, fold);
    p.z = boxFold(p.z, fold);
    return p;
}

float mandelboxDE(vec3 pos) {
    float scale = 1.0;
    float minRadius2 = 0.5;
    float fixedRadius2 = 1.5;
    float folding = 2.0;
    
    vec3 z = pos;
    vec3 offset = z;
    float dr = 1.0;
    
    for (int n = 0; n < MAX_ITERATIONS; n++) {
        // Box fold
        z = boxFold(z, folding);
        
        // Sphere fold
        float r2 = dot(z,z);
        if (r2 < minRadius2) { 
            float temp = fixedRadius2/minRadius2;
            z *= temp;
            dr *= temp;
        } else if (r2 < fixedRadius2) {
            float temp = fixedRadius2/r2;
            z *= temp;
            dr *= temp;
        }
        
        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0;
    }
    float r = length(z);
    return r/abs(dr);
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(DEL, 0.0);
    return normalize(vec3(
        mandelboxDE(p + e.xyy) - mandelboxDE(p - e.xyy),
        mandelboxDE(p + e.yxy) - mandelboxDE(p - e.yxy),
        mandelboxDE(p + e.yyx) - mandelboxDE(p - e.yyx)
    ));
}

float intersect(inout vec3 rO, inout vec3 rD, float epsilon) {
    float t = 0.0;
    float dist;
    
    for(int i = 0; i < 200; i++) {
        dist = mandelboxDE(rO);
        t += dist * 0.8;
        rO += rD * dist * 0.8;
        
        if(dist < epsilon || dot(rO, rO) > BOUNDING_RADIUS_2)
            break;
    }
    return dist;
}

vec3 blinn_phong(vec3 light, vec3 eye, vec3 pt, vec3 N) {
    vec3 baseColor = iMaterial;
    if (iUseObjTexture) {
        vec3 pp = pt * 0.5 + 0.5;
        baseColor = texture(iObjTexture, pp).rgb;
    }

    vec3 L = normalize(light - pt);
    vec3 E = -eye;
    vec3 H = normalize((E + L) / 2.0);

    float NdotL = dot(N, L);
    
    float ambiant = iAmbientLight;
    float diffuse = max(NdotL, 0.0) * iDiffuseScale;
    float spec = pow(max(dot(H, N), 0.0), iSpecularExp) * iSpecularScale;
    vec3 shade = baseColor * (ambiant + diffuse + spec);

    // gamma correction
    shade = pow(shade, vec3(1.0 / 2.2));

    return shade;
}

vec4 render(vec3 ro, vec3 rd, vec3 light, float epsilon) {
    const vec4 backgroundColor = vec4(0.0);
    vec4 color = backgroundColor;
    vec3 N = vec3(0.0);

    float dist = intersect(ro, rd, epsilon);
    out_distance_field = dist;

    if(dist < epsilon) {
        out_coord += ro;
        N = calcNormal(ro);
        color.xyz = blinn_phong(light, rd, ro, N);
        
        // Shadow calculation
        vec3 L = normalize(light - ro);
        vec3 p = ro + N * epsilon * 2.0;
        dist = intersect(p, L, epsilon);
        if(dist < epsilon)
            color.xyz *= 0.8 + (dot(L, N) + 1.0) * 0.1;
        
        color.w = 1.0;
    }
    else if (iUseBgTexture) {
        color.xyz = texture(iBgTexture, gl_FragCoord.xy / iResolution.xy).rgb;
        color.a = 1.0;
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

    vec3 lightSource = convertLocation(iLightSource.xy, 2 * iViewDistance);
    vec3 camera = convertLocation(iViewAngleXY.xy, iViewDistance);
    camera += iObjectOffset;
    vec3 target = vec3(0.0) + iObjectOffset;
    float roll = 0.0;
    float focal = iFocalPlane;
    mat3 cameraMat = getCameraMatrix(camera, target, roll);

    float mdim = max(res.x, res.y);
    int AA = max(1, iAntialias);
    vec4 color = vec4(0.0);
    vec2 xy = (2.0 * fragCoord - res.xy) / mdim;
    
    for (int i = 0; i < AA; i++)
        for (int j = 0; j < AA; j++) {
            vec2 p = xy + vec2(float(i), float(j)) / (2.0 * mdim);
            vec3 rd = normalize(cameraMat * vec3(p, focal));
            color += render(camera, rd, lightSource, 9e-3);
        }
    color /= float(AA * AA);
    out_coord /= float(AA * AA);
    
    out_normals = normalize(out_normals / float(AA * AA));
    out_color = color;
    out_debug = vec3(v_texcoord, 1.0);

    if (out_coord == vec3(0.0)) {
        out_depth = 10.0;
        out_object_id = -1;
    }
    else {
        vec3 cameraToCoord = camera - out_coord;
        vec3 viewDir = normalize(camera - target);
        out_depth = dot(cameraToCoord, viewDir) / (iViewDistance + 1.0);
        out_object_id = v_object_id;
    }
}
