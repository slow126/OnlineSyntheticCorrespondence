#version 430

in vec2 v_texcoord;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 4) out float out_depth;
layout(location = 5) out int out_object_id;
layout(location = 6) out float out_distance_field;
// Uniforms to access the renderbuffer textures
uniform sampler2D fbo_color;
uniform sampler2D fbo_coord;
uniform sampler2D fbo_normal;
uniform sampler2D fbo_debug;
uniform sampler2D fbo_depth;
uniform isampler2D fbo_object_id;
uniform sampler2D fbo_distance_field;

ivec2 res = ivec2(320, 320);

void main() {
    out_color = texelFetch(fbo_color, ivec2(v_texcoord * res), 0);
    out_coord = texelFetch(fbo_coord, ivec2(v_texcoord * res), 0).xyz;
    out_normals = texelFetch(fbo_normal, ivec2(v_texcoord * res), 0).xyz;
    out_debug = texelFetch(fbo_debug, ivec2(v_texcoord * res), 0).xyz;
    out_depth = texelFetch(fbo_depth, ivec2(v_texcoord * res), 0).r;
    out_object_id = texelFetch(fbo_object_id, ivec2(v_texcoord * res), 0).r;
    out_distance_field = texelFetch(fbo_distance_field, ivec2(v_texcoord * res), 0).r;

    float alpha = smoothstep(-fwidth(out_distance_field), fwidth(out_distance_field), out_distance_field);
    out_color = vec4(out_color.rgb * alpha, alpha);
    out_coord = out_coord * alpha;
    out_normals = out_normals * alpha;
    out_debug = out_debug;
    out_depth = mix(-1.0/0.0, out_depth, alpha);
    out_object_id = (alpha > 0.5) ? out_object_id : -1;
    out_distance_field = out_distance_field * alpha;
}