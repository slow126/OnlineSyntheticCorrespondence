#version 430

#define HW_PERFORMANCE 1

// view
uniform vec3      iResolution;
uniform float     iViewDistance;

// scene
uniform vec2      iLightSource    = vec2(0.0);

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 4) out float out_depth;

void main()
{
    // ADD these to top
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 uv = fragCoord/iResolution.xy;
    vec3 res = iResolution;

        
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