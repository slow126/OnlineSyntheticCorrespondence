/* 
Much of this code is adapted directly from the paper by Keenan Crane:
https://www.cs.cmu.edu/~kmcrane/Projects/QuaternionJulia/paper.pdf
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
// fractal
uniform vec4      iJuliaC;
// texture
uniform bool      iUseObjTexture  = false;
uniform sampler3D iObjTexture;
uniform bool      iUseBgTexture   = false;
uniform sampler2D iBgTexture;


#define ESCAPE_THRESHOLD 10.0
#define DEL 0.0001
#define BOUNDING_RADIUS_2 5.0

#define PI 3.14159265359

out vec4 out_color;
out vec3 out_coord;
out vec3 out_normals;

// Noise function for terrain generation
float hash(float n) {
    return fract(sin(n)*43758.5453123);
}

float noise(vec2 x) {
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    return mix(mix(hash(n), hash(n+1.0), f.x),
               mix(hash(n+57.0), hash(n+58.0), f.x), f.y);
}

float fbm(vec2 x) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100);
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < 5; ++i) {
        v += a * noise(x);
        x = rot * x * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

float terrainHeight(vec2 p) {
    return fbm(p * 3.0) * 0.5; // Scale the height as needed
}

vec3 terrainNormal(vec2 p) {
    float eps = 0.01;
    float h = terrainHeight(p);
    float dx = terrainHeight(p + vec2(eps, 0)) - h;
    float dy = terrainHeight(p + vec2(0, eps)) - h;
    return normalize(vec3(-dx/eps, -dy/eps, 1.0));
}

vec4 quatMult(vec4 q1, vec4 q2) {
    vec4 r;
    r.x = q1.x*q2.x - dot(q1.yzw, q2.yzw);
    r.yzw = q1.x*q2.yzw + q2.x*q1.yzw + cross(q1.yzw, q2.yzw);
    return r;
}

vec4 quatSq(vec4 q) {
    vec4 r;
    r.x = q.x*q.x - dot(q.yzw, q.yzw);
    r.yzw = 2.0*q.x*q.yzw;
    return r;
}

void julia(inout vec4 q, inout vec4 qp, vec4 c, int maxIterations) {
    for(int i=0; i<maxIterations; i++) {
        qp = 2.0 * quatMult(q, qp);
        q = quatSq(q) + c;
        if( dot( q, q ) > ESCAPE_THRESHOLD )
            break;
    }
}

vec3 normEstimate(vec3 p, vec4 c, int maxIterations) {
    vec4 qP = vec4(p, 0);
    vec4 gx1 = qP - vec4( DEL, 0, 0, 0 );
    vec4 gx2 = qP + vec4( DEL, 0, 0, 0 );
    vec4 gy1 = qP - vec4( 0, DEL, 0, 0 );
    vec4 gy2 = qP + vec4( 0, DEL, 0, 0 );
    vec4 gz1 = qP - vec4( 0, 0, DEL, 0 );
    vec4 gz2 = qP + vec4( 0, 0, DEL, 0 );
    for(int i = 0; i<maxIterations; i++) {
        gx1 = quatSq( gx1 ) + c;
        gx2 = quatSq( gx2 ) + c;
        gy1 = quatSq( gy1 ) + c;
        gy2 = quatSq( gy2 ) + c;
        gz1 = quatSq( gz1 ) + c;
        gz2 = quatSq( gz2 ) + c;
    }
    float gradX = length(gx2) - length(gx1);
    float gradY = length(gy2) - length(gy1);
    float gradZ = length(gz2) - length(gz1);
    vec3 N = normalize(vec3(gradX, gradY, gradZ));
    return N;
}

float intersect(inout vec3 rO, inout vec3 rD, vec4 c, int maxIterations, float epsilon) {
    float dist;
    
    int maxSteps = 200;
    float n;
    for (int i = 0; i < maxSteps; i++) {
        n = float(i);
        vec4 z = vec4(rO, 0);
        vec4 zp = vec4(1.0, vec3(0.0));
        julia(z, zp, c, maxIterations);
        float normZ = length(z);
        dist = 0.5 * normZ * log(normZ) / length(zp);
        rO += rD * dist;
        if( dist < epsilon || dot(rO, rO) > BOUNDING_RADIUS_2 )
            break;
    }
    return dist;
}

float intersectTerrain(vec3 ro, vec3 rd) {
    float t = 0.0;
    for(int i = 0; i < 128; i++) {
        vec3 p = ro + rd * t;
        float h = terrainHeight(p.xz);
        if(p.y < h) return t;
        t += 0.1;
    }
    return -1.0;
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

    shade[0] = pow(shade[0], 1.0 / 2.2);
    shade[1] = pow(shade[1], 1.0 / 2.2);
    shade[2] = pow(shade[2], 1.0 / 2.2);

    return shade;
}

vec4 render(vec3 ro, vec3 rd, vec4 c, vec3 light, int maxIter, float epsilon) {
    const vec4 backgroundColor = vec4(0.0);
    vec4 color = backgroundColor;
    vec3 N = vec3(0.0);

    // First check terrain intersection
    float terrainDist = intersectTerrain(ro, rd);
    float juliaSetDist = intersect(ro, rd, c, maxIter, epsilon);

    if(juliaSetDist < epsilon) {
        // Julia set rendering (elevated above terrain)
        ro.y += 1.0; // Lift the Julia set above the terrain
        out_coord += ro;
        N = normEstimate(ro, c, maxIter);
        color.xyz = blinn_phong(light, rd, ro, N);
        
        if(true) {
            vec3 L = normalize(light - ro);
            vec3 p = ro + N * epsilon * 2.;
            juliaSetDist = intersect(p, L, c, maxIter, epsilon);
            if(juliaSetDist < epsilon)
                color.xyz *= 0.8 + (dot(L, N) + 1.0) * 0.1;
        }
        color.w = 1.0;
    }
    else if(terrainDist > 0.0) {
        // Terrain rendering
        vec3 p = ro + rd * terrainDist;
        vec3 terrainN = terrainNormal(p.xz);
        color.xyz = vec3(0.5) * blinn_phong(light, rd, p, terrainN);
        color.w = 1.0;
    }
    else if (iUseBgTexture) {
        color.xyz = texture(iBgTexture, gl_FragCoord.xy / iResolution.xy).rgb;
        color.a = 1.0;
    }

    out_normals += N;
    return color;
}

mat3 getCameraMatrix(vec3 camera, vec3 target, float roll)
{
    vec3 u = vec3(sin(roll), cos(roll), 0.0);
    vec3 fw = normalize(target - camera);
    vec3 rt = normalize(cross(fw, u));
    vec3 up = normalize(cross(rt, fw));
    mat3 mat = mat3(rt, up, fw);
    return mat;
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
    vec4 c = iJuliaC;

    vec3 lightSource = convertLocation(iLightSource.xy, 2 * iViewDistance);
    vec3 camera = convertLocation(iViewAngleXY.xy, iViewDistance);
    vec3 target = vec3(0.0, 0.0, 0.0);
    float roll = 0.0;
    float focal = iFocalPlane;
    mat3 cameraMat = getCameraMatrix(camera, target, roll);

    float mdim = max(res.x, res.y);
    int AA = max(1, iAntialias);
    vec4 color;
    vec2 xy = (2.0 * fragCoord - res.xy) / mdim;
    for (int i = 0; i < AA; i++)
        for (int j = 0; j < AA; j++) {
            vec2 p = xy + vec2(float(i), float(j)) / (2.0 * mdim);
            vec3 rd = normalize(cameraMat * vec3(p, focal));
            color += render(camera, rd, c, lightSource, 10, 9e-3);
        }
    color /= float(AA * AA);
    out_coord /= float(AA * AA);

    out_normals = out_normals / float(AA * AA);
    float nlen = length(out_normals);
    out_normals = nlen > 0 ? out_normals / nlen : out_normals;

    out_color = color;
}