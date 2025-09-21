#version 430

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

in vec2 v_texcoord;
in flat int v_object_id;

#define MaximumRaySteps 100
#define MaximumDistance 1000.
#define MinimumDistance .01
#define PI 3.14159265359

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 4) out float out_depth;
layout(location = 5) out int out_object_id;

// SDF FUNCTIONS //

float SignedDistSphere(vec3 p, float s) {
    return length(p) - s;
}

float SignedDistBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

float SignedDistPlane(vec3 p, vec4 n) {
    return dot(p, n.xyz) + n.w;
}

float SignedDistRoundBox(in vec3 p, in vec3 b, in float r) {
    vec3 q = abs(p) - b;
    return min(max(q.x, max(q.y, q.z)), 0.0) + length(max(q, 0.0)) - r;
}

// BOOLEAN OPERATORS //

float opU(float d1, float d2) {
    return (d1 < d2) ? d1 : d2;
}

vec4 opS(vec4 d1, vec4 d2) {
    return (-d1.w > d2.w) ? -d1 : d2;
}

vec4 opI(vec4 d1, vec4 d2) {
    return (d1.w > d2.w) ? d1 : d2;
}

float pMod1(inout float p, float size) {
    float halfsize = size * 0.5;
    float c = floor((p + halfsize) / size);
    p = mod(p + halfsize, size) - halfsize;
    p = mod(-p + halfsize, size) - halfsize;
    return c;
}

// SMOOTH BOOLEAN OPERATORS //

float opUS(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    float dist = mix(d2, d1, h) - k * h * (1.0 - h);
    return dist;
}

vec4 opSS(vec4 d1, vec4 d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2.w + d1.w) / k, 0.0, 1.0);
    float dist = mix(d2.w, -d1.w, h) + k * h * (1.0 - h);
    vec3 color = mix(d2.rgb, d1.rgb, h);
    return vec4(color.rgb, dist);
}

vec4 opIS(vec4 d1, vec4 d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2.w - d1.w) / k, 0.0, 1.0);
    float dist = mix(d2.w, d1.w, h) + k * h * (1.0 - h);
    vec3 color = mix(d2.rgb, d1.rgb, h);
    return vec4(color.rgb, dist);
}

mat2 Rotate(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

vec3 R(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l - p),
         r = normalize(cross(vec3(0, 1, 0), f)),
         u = cross(f, r),
         c = p + f * z,
         i = c + uv.x * r + uv.y * u,
         d = normalize(i - p);
    return d;
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float map(float value, float min1, float max1, float min2, float max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

float sierpinski3(vec3 z) {
    float iterations = 25.0;
    float Scale = 2.0 + (sin(5.0 / 2.0) + 1.0);
    vec3 Offset = 3.0 * vec3(1.0, 1.0, 1.0);
    float bailout = 1000.0;

    float r = length(z);
    int n = 0;
    while (n < int(iterations) && r < bailout) {
        z.x = abs(z.x);
        z.y = abs(z.y);
        z.z = abs(z.z);

        if (z.x - z.y < 0.0) z.xy = z.yx;
        if (z.x - z.z < 0.0) z.xz = z.zx;
        if (z.y - z.z < 0.0) z.zy = z.yz;

        z.x = z.x * Scale - Offset.x * (Scale - 1.0);
        z.y = z.y * Scale - Offset.y * (Scale - 1.0);
        z.z = z.z * Scale;

        if (z.z > 0.5 * Offset.z * (Scale - 1.0)) {
            z.z -= Offset.z * (Scale - 1.0);
        }

        r = length(z);
        n++;
    }

    return (length(z) - 2.0) * pow(Scale, -float(n));
}

float DistanceEstimator(vec3 p) {
    p.yz *= Rotate(0.2 * PI);
    p.yx *= Rotate(0.3 * PI);
    p.xz *= Rotate(0.29 * PI);
    float sierpinski = sierpinski3(p);
    return sierpinski;
}

vec4 RayMarcher(vec3 ro, vec3 rd) {
    float steps = 0.0;
    float totalDistance = 0.0;
    float minDistToScene = 100.0;
    vec3 minDistToScenePos = ro;
    float minDistToOrigin = 100.0;
    vec3 minDistToOriginPos = ro;
    vec4 col = vec4(0.0, 0.0, 0.0, 1.0);
    vec3 curPos = ro;
    bool hit = false;
    vec3 p = vec3(0.0);

    for (steps = 0.0; steps < float(MaximumRaySteps); steps++) {
        p = ro + totalDistance * rd;
        float distance = DistanceEstimator(p);
        curPos = ro + rd * totalDistance;
        if (minDistToScene > distance) {
            minDistToScene = distance;
            minDistToScenePos = curPos;
        }
        if (minDistToOrigin > length(curPos)) {
            minDistToOrigin = length(curPos);
            minDistToOriginPos = curPos;
        }
        totalDistance += distance;
        if (distance < MinimumDistance) {
            hit = true;
            break;
        }
        else if (distance > MaximumDistance) {
            break;
        }
        out_coord = p;
    }

    float iterations = float(steps) + log(log(MaximumDistance)) / log(2.0) - log(log(dot(curPos, curPos))) / log(2.0);

    if (hit) {
        col.rgb = vec3(0.8 + (length(curPos) / 8.0), 1.0, 0.8);
        col.rgb = hsv2rgb(col.rgb);
    }
    else {
        col.rgb = vec3(0.8 + (length(minDistToScenePos) / 8.0), 1.0, 0.8);
        col.rgb = hsv2rgb(col.rgb);
        col.rgb *= 1.0 / pow(minDistToScene, 1.0);
        col.rgb /= 15.0;
    }
    col.rgb /= iterations / 10.0;
    col.rgb /= pow(distance(ro, minDistToScenePos), 2.0);
    col.rgb *= 2000.0;

    return col;
}

vec3 convertLocation(vec2 offsets, float distance) {
    float theta = PI * offsets.x; // rotation around the y axis
    float phi = -PI * offsets.y;  // rotation around the x axis

    float cx = cos(phi), sx = sin(phi);
    float cy = cos(theta), sy = sin(theta);
    
    vec3 location = distance * vec3(sy * cx, -sx, cy * cx);
    return location;
}

mat3 getCameraMatrix(vec3 camera, vec3 target, float roll) {
    vec3 u = vec3(sin(roll), cos(roll), 0.0); // world-space up vector
    vec3 fw = normalize(target - camera);     // forward vector
    vec3 rt = normalize(cross(fw, u));        // right vector
    vec3 up = normalize(cross(rt, fw));       // camera-space up vector
    mat3 mat = mat3(rt, up, fw);
    return mat;
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    uv *= 0.2;
    uv.y -= 0.015;

    vec3 ro = convertLocation(iViewAngleXY.xy, iViewDistance);
    vec3 target = vec3(0.0);
    float roll = 0.0;
    mat3 cameraMat = getCameraMatrix(ro, target, roll);
    
    vec3 rd = normalize(cameraMat * vec3(uv * 2.0 - 1.0, iFocalPlane));
    out_debug = rd;

    vec4 col = RayMarcher(ro, rd);
    
    out_color = col;
    out_debug = vec3(v_texcoord, 1.0);

    if (out_coord == vec3(0.0)) {
        out_depth = 10.0;
        out_object_id = 0;
    }
    else {
        vec3 cameraToCoord = ro - out_coord;
        vec3 viewDir = normalize(ro - target);
        out_depth = dot(cameraToCoord, viewDir);
        out_object_id = v_object_id;
    }
}