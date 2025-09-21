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

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    
    // Camera setup
    float cameraHeight = 3.0;
    float cameraSpeed = 0.3;
    vec3 cameraPos = vec3(iTime * cameraSpeed, cameraHeight, iTime * cameraSpeed);
    vec3 rayDir = normalize(vec3(uv * 2.0 - 1.0, 1.0));
    
    // Camera angle
    float pitch = -0.3;
    rayDir.yz *= mat2(cos(pitch), -sin(pitch), sin(pitch), cos(pitch));
    
    // Ray marching parameters
    float maxDist = 100.0;
    float dist = 0.0;
    float minStep = 0.1;
    vec3 p;
    
    // Generate base terrain
    for(int i = 0; i < 120; i++) {
        p = cameraPos + rayDir * dist;
        
        // Terrain height
        float terrainHeight = 0.0;
        float scale = 2.0;
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
    } else {
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
        
        // Add distance fog
        float fog = 1.0 - exp(-dist * 0.03);
        color = mix(color, skyColor, fog);
    }
    
    // Final color adjustment
    color = pow(color, vec3(0.8)); // Gamma correction
    fragColor = vec4(color, 1.0);
}