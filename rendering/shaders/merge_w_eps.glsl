#version 430

in vec2 v_texcoord;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 4) out float out_depth;
layout(location = 5) out int out_object_id;

// Uniforms to access the renderbuffer textures
uniform sampler2D fbo1_color;
uniform sampler2D fbo1_coord;
uniform sampler2D fbo1_normal;
uniform sampler2D fbo1_debug;
uniform sampler2D fbo1_depth;
uniform isampler2D fbo1_object_id;
uniform sampler2D fbo2_color;
uniform sampler2D fbo2_coord;
uniform sampler2D fbo2_normal;
uniform sampler2D fbo2_debug;
uniform sampler2D fbo2_depth;
uniform isampler2D fbo2_object_id;

ivec2 res = ivec2(320, 320);

void main() {
    // Sample data from both FBOs
    vec4 color1 = texture(fbo1_color, v_texcoord);
    vec3 coord1 = texture(fbo1_coord, v_texcoord).xyz;
    vec3 normal1 = texture(fbo1_normal, v_texcoord).xyz;
    vec3 debug1 = texture(fbo1_debug, v_texcoord).xyz;
    float depth1 = texelFetch(fbo1_depth, ivec2(v_texcoord * res), 0).r;
    int object_id1 = texture(fbo1_object_id, v_texcoord).r;
    vec4 color2 = texture(fbo2_color, v_texcoord);
    vec3 coord2 = texture(fbo2_coord, v_texcoord).xyz;
    vec3 normal2 = texture(fbo2_normal, v_texcoord).xyz;
    vec3 debug2 = texture(fbo2_debug, v_texcoord).xyz;
    float depth2 = texelFetch(fbo2_depth, ivec2(v_texcoord * res), 0).r;
    int object_id2 = texture(fbo2_object_id, v_texcoord).r;

    // Check validity
    bool valid1 = (depth1 < 10.9999);
    bool valid2 = (depth2 < 10.9999);

    if (!valid1 && !valid2) {
        discard;
    } else if (!valid1) {
        out_color = color2;
        out_coord = coord2;
        out_normals = normal2;
        out_debug = debug2;
        out_depth = depth2;
        out_object_id = object_id2;
    } else if (!valid2) {
        out_color = color1;
        out_coord = coord1;
        out_normals = normal1;
        out_debug = debug1;
        out_depth = depth1;
        out_object_id = object_id1;
    } else {
        // Both fragments are valid
        float epsilon = 0.01; // Base epsilon
        float bias = 0.02;   // Additional bias for FBO2
        
        // Calculate view direction for both fragments
        vec3 viewDir1 = normalize(-coord1);
        vec3 viewDir2 = normalize(-coord2);
        
        // Calculate depth bias based on surface orientation
        float bias1 = max(0.01 * (1.0 - dot(normal1, viewDir1)), 0.001);
        float bias2 = max(0.01 * (1.0 - dot(normal2, viewDir2)), 0.001) - bias; // Subtract bias to favor FBO2
        
        // Compare distances with bias
        float dist1 = length(coord1) + bias1;
        float dist2 = length(coord2) + bias2;

        // Add a small neighborhood check to prevent fragments from different objects intersecting
        float depthDiff = abs(dist1 - dist2);
        
        if (depthDiff < epsilon) {
            // If depths are very close, prefer FBO2
            out_color = color2;
            out_coord = coord2;
            out_normals = normal2;
            out_debug = debug2;
            out_depth = depth2;
            out_object_id = object_id2;
        } else {
            // Normal depth test with bias
            if (dist1 < dist2) {
                out_color = color1;
                out_coord = coord1;
                out_normals = normal1;
                out_debug = debug1;
                out_depth = depth1;
                out_object_id = object_id1;
            } else {
                out_color = color2;
                out_coord = coord2;
                out_normals = normal2;
                out_debug = debug2;
                out_depth = depth2;
                out_object_id = object_id2;
            }
        }
    }
}