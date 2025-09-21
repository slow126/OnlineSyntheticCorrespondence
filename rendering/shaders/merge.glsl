#version 430

in vec2 v_texcoord;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 4) out float out_depth;
layout(location = 5) out int out_object_id;
layout(location = 6) out float out_distance_field;
layout(location = 7) out vec2 out_uv;
// Uniforms to access the renderbuffer textures
uniform sampler2D fbo1_color;
uniform sampler2D fbo1_coord;
uniform sampler2D fbo1_normal;
uniform sampler2D fbo1_debug;
uniform sampler2D fbo1_depth;
uniform isampler2D fbo1_object_id;
uniform sampler2D fbo1_distance_field;
uniform sampler2D fbo1_uv;
uniform sampler2D fbo2_color;
uniform sampler2D fbo2_coord;
uniform sampler2D fbo2_normal;
uniform sampler2D fbo2_debug;
uniform sampler2D fbo2_depth;
uniform isampler2D fbo2_object_id;
uniform sampler2D fbo2_distance_field;
uniform sampler2D fbo2_uv;
ivec2 res = ivec2(320, 320);

void main() {
    // Sample data from FBO 1
    vec4 color1 = texture(fbo1_color, v_texcoord);
    vec3 coord1 = texelFetch(fbo1_coord, ivec2(v_texcoord * res), 0).xyz;
    vec3 normal1 = texelFetch(fbo1_normal, ivec2(v_texcoord * res), 0).xyz;
    vec3 debug1 = texture(fbo1_debug, v_texcoord).xyz;
    float depth1 = texelFetch(fbo1_depth, ivec2(v_texcoord * res), 0).r;
    int object_id1 = texelFetch(fbo1_object_id, ivec2(v_texcoord * res), 0).r;
    float distance_field1 = texelFetch(fbo1_distance_field, ivec2(v_texcoord * res), 0).r;
    vec2 uv1 = texelFetch(fbo1_uv, ivec2(v_texcoord * res), 0).xy;
    // Sample data from FBO 2
    vec4 color2 = texture(fbo2_color, v_texcoord);
    vec3 coord2 = texelFetch(fbo2_coord, ivec2(v_texcoord * res), 0).xyz;
    vec3 normal2 = texelFetch(fbo2_normal, ivec2(v_texcoord * res), 0).xyz;
    vec3 debug2 = texture(fbo2_debug, v_texcoord).xyz;
    float depth2 = texelFetch(fbo2_depth, ivec2(v_texcoord * res), 0).r;
    int object_id2 = texelFetch(fbo2_object_id, ivec2(v_texcoord * res), 0).r;
    float distance_field2 = texelFetch(fbo2_distance_field, ivec2(v_texcoord * res), 0).r;
    vec2 uv2 = texelFetch(fbo2_uv, ivec2(v_texcoord * res), 0).xy;

    // Depth test: Choose the fragment with the smaller depth value (closer to the camera)
    if (depth1 < depth2) {
        out_color = color1;
        // If you need to output other buffers, do it here as well:
        out_coord = coord1; // Example for writing to a second render target
        out_normals = normal1;
        out_debug = debug1;
        out_depth = depth1;
        out_object_id = object_id1;
        out_distance_field = distance_field1;
        out_uv = uv1;
    } else {
        out_color = color2;
        // If you need to output other buffers:
        out_coord = coord2;
        out_normals = normal2;
        out_debug = debug2;
        out_depth = depth2;
        out_object_id = object_id2;
        out_distance_field = distance_field2;
        out_uv = uv2;
    }
    
    
    // Use buffer1 outputs

    // out_color = color1;
    // out_coord = coord1;
    // out_normals = normal1;
    // out_debug = debug1;
    // out_depth = depth1;
    // out_object_id = texelFetch(fbo1_object_id, ivec2(v_texcoord * res), 0).r;
    
    // // Prevent buffer2 from being optimized out by using the values
    // vec4 dummy = color2 * 0.01;
    // dummy += vec4(coord2 * 0.01, 0.0); 
    // dummy += vec4(normal2 * 0.01, 0.0);
    // dummy += vec4(debug2 * 0.01, 0.0);
    // dummy += vec4(depth2 * 0.01);
    // dummy += vec4(float(texelFetch(fbo2_object_id, ivec2(v_texcoord * res), 0).r) * 0.01);
    // out_color += dummy * 0.01; // Add dummy without affecting output
}