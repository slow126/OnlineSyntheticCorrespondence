#version 430
uniform mat4 u_projection;  // Projection matrix
uniform mat4 u_model_view;  // Model-view matrix

in vec3 in_vert;  // Vertex position
void main() {
    vec4 world_position = u_model_view * vec4(in_vert, 1.0);  // Transform to world space
    gl_Position = u_projection * world_position;             // Transform to clip space
}
