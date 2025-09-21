#version 430

in vec2 v_texcoord;
out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 4) out float out_depth;
layout(location = 5) out int out_object_id;
layout(location = 6) out float out_distance_field;

// Number of active objects to process
uniform int num_active_objects;

// Arrays for each texture type, supporting up to 10 objects
uniform sampler2D color_textures[10];
uniform sampler2D coord_textures[10];
uniform sampler2D normal_textures[10];
uniform sampler2D debug_textures[10];
uniform sampler2D depth_textures[10];
uniform isampler2D object_id_textures[10];
uniform sampler2D distance_field_textures[10];

// Force the compiler to keep all uniforms by accessing them
// This prevents optimization from removing "unused" uniforms
vec4 force_keep_uniforms() {
    vec4 dummy = vec4(0.0);
    for (int i = 0; i < 10; i++) {
        dummy += texture(color_textures[i], vec2(0.0)) * 0.0;
        dummy += vec4(texture(coord_textures[i], vec2(0.0)).xyz, 0.0) * 0.0;
        dummy += vec4(texture(normal_textures[i], vec2(0.0)).xyz, 0.0) * 0.0;
        dummy += vec4(texture(debug_textures[i], vec2(0.0)).xyz, 0.0) * 0.0;
        dummy += texture(depth_textures[i], vec2(0.0)) * 0.0;
        dummy += vec4(float(texture(object_id_textures[i], vec2(0.0)).r), 0.0, 0.0, 0.0) * 0.0;
        dummy += texture(distance_field_textures[i], vec2(0.0)) * 0.0;
    }
    return dummy;
}

void main() {
    // Call the function but don't use the result
    // This ensures the compiler can't optimize away the uniforms
    vec4 dummy = force_keep_uniforms();
    
    ivec2 res = textureSize(depth_textures[0], 0);
    ivec2 pixel = ivec2(v_texcoord * res);
    
    // Start with maximum depth and no object
    float min_depth = 1000.0;
    int closest_index = -1;
    
    // Find closest object (with smallest depth value)
    for (int i = 0; i < num_active_objects; i++) {
        float depth = texelFetch(depth_textures[i], pixel, 0).r;
        
        // Skip if depth is 1.0 (no object/background)
        if (depth < min_depth) {
            min_depth = depth;
            closest_index = i;
        }
    }
    
    // // If no valid object found, output default/empty values
    // if (closest_index == -1) {
    //     out_color = vec4(0.0, 0.0, 0.0, 1.0);
    //     out_coord = vec3(0.0);
    //     out_normals = vec3(0.0);
    //     out_debug = vec3(0.0);
    //     out_depth = 1.0;
    //     out_object_id = -1;
    //     out_distance_field = 1000.0;
    //     return;
    // }
    
    // Output values from the closest object
    out_color = texture(color_textures[closest_index], v_texcoord);
    out_coord = texelFetch(coord_textures[closest_index], pixel, 0).xyz;
    out_normals = texelFetch(normal_textures[closest_index], pixel, 0).xyz;
    out_debug = texture(debug_textures[closest_index], v_texcoord).xyz;
    out_depth = texelFetch(depth_textures[closest_index], pixel, 0).r;
    out_object_id = texelFetch(object_id_textures[closest_index], pixel, 0).r;
    out_distance_field = texelFetch(distance_field_textures[closest_index], pixel, 0).r;
}
