#version 430

#define HW_PERFORMANCE 1

// view
uniform vec3      iResolution;
uniform float     iFocalPlane;
uniform float     iViewDistance;
uniform vec2      iViewAngleXY; 

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


layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 4) out float out_depth;

in vec2 v_texcoord;

#define PI 3.14159265359
#define NUM_OCTAVES 10

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

vec3 convertLocation(vec2 offsets, float distance) {
    float theta = PI * offsets.x; // rotation around the y axis
    float phi = -PI * offsets.y;  // rotation around the x axis

    float cx = cos(phi), sx = sin(phi);
    float cy = cos(theta), sy = sin(theta);
    
    vec3 location = distance * vec3(sy * cx, -sx, cy * cx);
    return location;
}

mat3 getCameraMatrix(vec3 camera, vec3 target, float roll)
{
    vec3 u = vec3(sin(roll), cos(roll), 0.0); // world-space up vector
    vec3 fw = normalize(target - camera);     // forward vector
    vec3 rt = normalize(cross(fw, u));        // right vector
    vec3 up = normalize(cross(rt, fw));       // camera-space up vector
    mat3 mat = mat3(rt, up, fw);
    return mat;
}

vec3 normEstimate(vec3 p, int maxIterations) {
    vec4 qP = vec4(p, 0);
    vec4 gx1 = qP - vec4( DEL, 0, 0, 0 );
    vec4 gx2 = qP + vec4( DEL, 0, 0, 0 );
    vec4 gy1 = qP - vec4( 0, DEL, 0, 0 );
    vec4 gy2 = qP + vec4( 0, DEL, 0, 0 );
    vec4 gz1 = qP - vec4( 0, 0, DEL, 0 );
    vec4 gz2 = qP + vec4( 0, 0, DEL, 0 );

    // Add distance to the surface
    

    float gradX = length(gx2) - length(gx1);
    float gradY = length(gy2) - length(gy1);
    float gradZ = length(gz2) - length(gz1);
    vec3 N = normalize(vec3(gradX, gradY, gradZ));
    return N;
}

void main()
{
    // ADD these to top
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 uv = fragCoord/iResolution.xy;
    vec3 res = iResolution;


    vec3 lightSource = convertLocation(iLightSource.xy, 2 * iViewDistance);
    // camera
    vec3 cameraPos = convertLocation(iViewAngleXY.xy, iViewDistance);
	vec3 target = vec3(0.0, 0.0, 0.0);
    float roll = 0.0; // camera roll
    float focal = iFocalPlane; // focal plane (zoom)
    mat3 cameraMat = getCameraMatrix(cameraPos, target, roll);

    float mdim = max(res.x, res.y);
    vec2 p_tmp = fragCoord;
    vec3 rayDir = normalize(cameraMat * vec3(p_tmp, focal));

    out_debug = rayDir;

    // Ray marching parameters
    float maxDist = 200.0;
    float dist = 0.0;
    float minStep = 0.1;
    vec3 p;
    
    // Generate base terrain with -2 y offset
    float yOffset = -2.0;
    
    for(int i = 0; i < 2000; i++) {
        p = cameraPos + rayDir * dist;
        
        // Terrain height with offset
        float terrainHeight = yOffset;
        float scale = 1.0;
        float amplitude = 1.0;
        
        // Fractal terrain
        vec2 worldPos = p.xz;
        for(int j = 0; j < 5; j++) {
            terrainHeight += amplitude * noise(worldPos * scale);
            scale *= 2.0;
            amplitude *= 0.5;
        }
        
        // Add smaller detail
        terrainHeight += noise(worldPos * 8.0) * 0.1;
        
        float h = p.y - terrainHeight;
        if(h < minStep) break;
        
        dist += max(h * 0.5, minStep);
        if(dist > maxDist) break;
    }
    
    // Color calculation
    vec3 skyColor = vec3(0.6, 0.8, 1.0);
    vec3 grassColor = vec3(0.3, 0.5, 0.2);
    vec3 rockColor = vec3(0.4, 0.3, 0.2);
    vec3 flowerColor = vec3(0.9, 0.5, 0.4);
    
    vec3 color;
    if(dist > maxDist) {
        // Sky color with atmospheric scattering
        color = skyColor;
        float sunStrength = pow(max(0.0, dot(rayDir, normalize(vec3(0.5, 0.4, 0.3)))), 32.0);
        color += vec3(1.0, 0.7, 0.3) * sunStrength;
        out_coord = vec3(0.0); // No collision for sky
        out_depth = maxDist;

    } else {
        // Store collision coordinates
        out_coord = p;
        out_depth = dist;
        // Terrain coloring
        float height = noise(p.xz * 2.0);
        
        // Base terrain color
        color = mix(grassColor, rockColor, smoothstep(0.4, 0.7, height));
        
        // Add vegetation detail
        float vegetation = noise(p.xz * 15.0);
        float flowers = step(0.95, noise(p.xz * 20.0));
        
        // Mix in vegetation and flowers
        color = mix(color, grassColor * 1.2, vegetation * 0.3);
        color = mix(color, flowerColor, flowers * 0.7);
        
        // Add trees based on noise
        float treeDensity = noise(p.xz * 0.5);
        if(treeDensity > 0.5) {
            vec2 treePos = floor(p.xz); // Round tree position to nearest integer
            float treeHeight = 1.0 + noise(treePos) * 0.5;
            float treeRadius = 0.5;
            
            // Check if current position is within tree oval
            vec2 treeRelPos = p.xz - treePos;
            float treeShape = length(treeRelPos / vec2(treeRadius, treeHeight));
            if(treeShape < 1.0) {
                color = mix(color, vec3(0.2, 0.5, 0.1), 0.6);
            }
        }
    }
    
    // Final color adjustment
    color = pow(color, vec3(0.8)); // Gamma correction
    out_color = vec4(color, 1.0);
    //TODO: Replace with acutal normals. 
    out_normals = out_coord;
    out_depth = out_depth / maxDist;
    out_coord = out_coord;

    // Add to prevent the compiler from optimizing out the texcoord
    vec2 texcoord = v_texcoord;
    out_debug = vec3(texcoord, 1.0);
}