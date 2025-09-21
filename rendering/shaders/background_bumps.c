/* 
Shader for rendering a bumpy terrain background
Using same interface as quatJulia.c but generating terrain instead of fractals
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

#define PI 3.14159265359
#define NUM_OCTAVES 10

out vec4 out_color;
out vec3 out_coord;
out vec3 out_normals;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float noise(vec2 x) {
    vec2 f = fract(x);
    vec2 u = f*f*(3.0-2.0*f);
    vec2 p = floor(x);
    
    float a = random(p + vec2(0.0, 0.0));
    float b = random(p + vec2(1.0, 0.0));
    float c = random(p + vec2(0.0, 1.0));
    float d = random(p + vec2(1.0, 1.0));
    
    return a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y;
}

float fbm(vec2 x) {
    float g = 0.5;
    float f = 1.0;
    float a = 1.0;
    float t = 0.0;
    for(int i = 0; i < NUM_OCTAVES; ++i) {
        t += a * noise(f * x);
        f *= 2.0;
        a *= g;
    }
    return t;
}

vec3 terrainNormal(vec2 pos) {
    float eps = 0.001;
    float center = fbm(pos);
    float right = fbm(pos + vec2(eps, 0));
    float up = fbm(pos + vec2(0, eps));
    
    return normalize(vec3(
        (center - right) / eps,
        (center - up) / eps,
        1.0
    ));
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
    
    float ambient = iAmbientLight;
    float diffuse = max(NdotL, 0.0) * iDiffuseScale;
    float spec = pow(max(dot(H, N), 0.0), iSpecularExp) * iSpecularScale;
    vec3 shade = baseColor * (ambient + diffuse + spec);

    // Add snow on higher elevations
    float height = fbm(pt.xz);
    vec3 snowColor = vec3(1.0);
    shade = mix(shade, snowColor, smoothstep(0.7, 1.2, height));

    // Add distance fog
    float dist = length(eye);
    vec3 skyColor = vec3(0.6, 0.8, 1.0);
    float fog = 1.0 - exp(-dist * 0.03);
    shade = mix(shade, skyColor, fog);

    shade = pow(shade, vec3(1.0 / 2.2)); // gamma correction
    return shade;
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
    
    return distance * vec3(sy * cx, -sx, cy * cx);
}

vec4 render(vec3 ro, vec3 rd, vec3 light) {
    // Ray-plane intersection
    float t = -ro.y / rd.y;
    
    if(t < 0.0) return vec4(0.0);
    
    vec3 pos = ro + t * rd;
    vec2 p = pos.xz * 0.5;
    
    // Generate height using fbm
    float height = -0.5 + fbm(p) * 0.3;
    vec3 worldPos = vec3(pos.x, height, pos.z);
    
    vec3 N = terrainNormal(p);
    vec3 color = blinn_phong(light, rd, worldPos, N);
    
    out_coord += worldPos;
    out_normals += N;
    
    return vec4(color, 1.0);
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec3 res = iResolution;

    vec3 lightSource = convertLocation(iLightSource.xy, 2 * iViewDistance);
    vec3 camera = convertLocation(iViewAngleXY.xy, iViewDistance);
    vec3 target = vec3(0.0, 0.0, 0.0);
    float roll = 0.0;
    float focal = iFocalPlane;
    mat3 cameraMat = getCameraMatrix(camera, target, roll);

    float mdim = max(res.x, res.y);
    int AA = max(1, iAntialias);
    vec4 color = vec4(0.0);
    vec2 xy = (2.0 * fragCoord - res.xy) / mdim;
    
    for(int i = 0; i < AA; i++)
        for(int j = 0; j < AA; j++) {
            vec2 p = xy + vec2(float(i), float(j)) / (2.0 * mdim);
            vec3 rd = normalize(cameraMat * vec3(p, focal));
            color += render(camera, rd, lightSource);
        }
    
    color /= float(AA * AA);
    out_coord /= float(AA * AA);
    
    out_normals = normalize(out_normals / float(AA * AA));
    
    out_color = color;
}
