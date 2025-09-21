#!/usr/bin/env python3
import argparse
import os
import numpy as np
import moderngl
import matplotlib.pyplot as plt
from PIL import Image
from src.data.synth.texture import multi_texturing as texturing

def setup_moderngl(shader_path, resolution):
    """Set up ModernGL context and compile shader"""
    ctx = moderngl.create_standalone_context(backend='egl', require=430, device_index=0)

    # Read shader code
    with open(shader_path, 'r') as f:
        fragment_shader = f.read()

    # Basic vertex shader
    vertex_shader = (
        '#version 430\n'
        'in vec2 in_vert;\n'
        'in vec2 in_texcoord;\n'
        'out vec2 v_texcoord;\n'
        'void main() {\n'
        '    gl_Position = vec4(in_vert, 0.0, 1.0);\n'
        '    v_texcoord = in_texcoord;\n'
        '}\n'
    )

    # Create program
    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader
    )

    # Create vertex buffer with position and texture coordinates
    vertices = np.array([
        -1.0, -1.0,  0.0, 0.0,  # Bottom-left
         1.0, -1.0,  1.0, 0.0,  # Bottom-right
        -1.0,  1.0,  0.0, 1.0,  # Top-left
         1.0,  1.0,  1.0, 1.0,  # Top-right
    ], dtype=np.float32)

    indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

    vbo = ctx.buffer(vertices)
    ibo = ctx.buffer(indices)
    vao = ctx.vertex_array(
        program,
        [(vbo, '2f 2f', 'in_vert', 'in_texcoord')],
        ibo
    )

    # Create framebuffer with renderbuffers
    rbuf_col = ctx.renderbuffer(resolution, components=4)
    rbuf_coord = ctx.renderbuffer(resolution, components=3, dtype='f4')
    rbuf_normal = ctx.renderbuffer(resolution, components=3, dtype='f4')
    rbuf_debug = ctx.renderbuffer(resolution, components=3, dtype='f4')
    rbuf_depth = ctx.renderbuffer(resolution, components=1, dtype='f4')
    rbuf_object_id = ctx.renderbuffer(resolution, components=1, dtype='i4')
    rbuf_distance_field = ctx.renderbuffer(resolution, components=1, dtype='f4')

    fbo = ctx.framebuffer(
        color_attachments=[rbuf_col, rbuf_coord, rbuf_normal, rbuf_debug, rbuf_depth, rbuf_object_id, rbuf_distance_field]
    )

    return {
        'ctx': ctx,
        'fbo': fbo,
        'program': program,
        'vao': vao
    }

def update_uniforms(program, uniforms=None, **kw):
    """Update shader uniforms"""
    if uniforms is None:
        uniforms = {}
    uniforms.update(kw)
    for k, v in uniforms.items():
        if program.get(k, None) is not None:
            program[k] = v

def get_rendered(state, resolution):
    """Render the shader and get the output buffers"""
    state['fbo'].use()
    state['vao'].render(mode=4)  # TRIANGLES = 4
    
    # Read color attachments
    coord = np.frombuffer(state['fbo'].read(attachment=1, components=3, dtype='f4'), 
                         dtype=np.float32).reshape(resolution + (3,))
    normals = np.frombuffer(state['fbo'].read(attachment=2, components=3, dtype='f4'), 
                           dtype=np.float32).reshape(resolution + (3,))
    
    # Flip vertically (OpenGL has origin at bottom-left)
    coord = np.ascontiguousarray(coord[::-1])
    normals = np.ascontiguousarray(normals[::-1])
    
    return coord, normals

def get_color_output(state, resolution):
    """Get the color output from the framebuffer"""
    color = np.frombuffer(state['fbo'].read(attachment=0, components=4), 
                         dtype=np.uint8).reshape(resolution + (4,))
    color = np.ascontiguousarray(color[::-1])
    return color

def save_outputs(outputs, output_dir, base_filename):
    """Save outputs to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save color output
    color_img = Image.fromarray(outputs['color'][..., :3])
    color_img.save(os.path.join(output_dir, f"{base_filename}_color.png"))
    
    # Save geometry (normalized for visualization)
    geom = outputs['geometry']
    geom_norm = (geom - geom.min()) / (geom.max() - geom.min())
    geom_img = Image.fromarray((geom_norm * 255).astype(np.uint8))
    geom_img.save(os.path.join(output_dir, f"{base_filename}_geometry.png"))
    
    # Save normals (normalized from -1,1 to 0,1)
    normals = (outputs['normals'] + 1) / 2
    normals_img = Image.fromarray((normals * 255).astype(np.uint8))
    normals_img.save(os.path.join(output_dir, f"{base_filename}_normals.png"))
    
    print(f"Saved outputs to {output_dir}")

def display_outputs(outputs):
    """Display outputs using matplotlib"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display color
    axes[0].imshow(outputs['color'][..., :3])
    axes[0].set_title('Color Output')
    axes[0].axis('off')
    
    # Display geometry
    geom_norm = (outputs['geometry'] - outputs['geometry'].min()) / (outputs['geometry'].max() - outputs['geometry'].min())
    axes[1].imshow(geom_norm)
    axes[1].set_title('Geometry')
    axes[1].axis('off')
    
    # Display normals
    normals = (outputs['normals'] + 1) / 2  # Convert from [-1,1] to [0,1]
    axes[2].imshow(normals)
    axes[2].set_title('Normals')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run a GLSL shader and visualize the output')
    parser.add_argument('shader_path', help='Path to the fragment shader file')
    parser.add_argument('--resolution', type=int, nargs=2, default=[320, 320], help='Resolution (width height)')
    parser.add_argument('--save', action='store_true', help='Save outputs to files')
    parser.add_argument('--output-dir', default='shader_outputs', help='Directory to save outputs')
    parser.add_argument('--no-display', action='store_true', help='Do not display outputs')
    
    # Add common shader uniforms as arguments
    parser.add_argument('--view-distance', type=float, default=3.0, help='View distance (iViewDistance)')
    parser.add_argument('--view-angle', type=float, nargs=2, default=[0.0, 0.0], help='View angle XY (iViewAngleXY)')
    parser.add_argument('--focal-plane', type=float, default=1.0, help='Focal plane (iFocalPlane)')
    parser.add_argument('--antialias', type=int, default=1, help='Antialiasing samples (iAntialias)')
    parser.add_argument('--julia-c', type=float, nargs=4, default=[1.0, 0.3, 0.5, 0.6], help='Julia set parameter (iJuliaC)')
    
    args = parser.parse_args()
    
    # Set up resolution
    resolution = tuple(args.resolution)
    
    # Set up ModernGL
    state = setup_moderngl(args.shader_path, resolution)
    
    # Set up uniforms
    uniforms = {
        'iResolution': (float(resolution[0]), float(resolution[1]), 1.0),
        'iViewDistance': args.view_distance,
        'iViewAngleXY': args.view_angle,
        'iFocalPlane': args.focal_plane,
        'iAntialias': args.antialias,
        'iJuliaC': args.julia_c,
        'iLightSource': [0.0, 0.0],
        'iAmbientLight': 0.3,
        'iDiffuseScale': 0.3,
        'iSpecularScale': 0.8,
        'iSpecularExp': 50,
        'iObjectID': 0,
        'iObjectOffset': [0.0, 0.0, 0.0],
        'iMandelbulbP': 8.0
    }
    
    # Update uniforms
    update_uniforms(state['program'], uniforms)
    
    # Render
    geometry, normals = get_rendered(state, resolution)
    color = get_color_output(state, resolution)
    
    # Collect outputs
    outputs = {
        'color': color,
        'geometry': geometry,
        'normals': normals
    }
    
    # Save outputs if requested
    if args.save:
        base_filename = os.path.splitext(os.path.basename(args.shader_path))[0]
        save_outputs(outputs, args.output_dir, base_filename)
    
    # Display outputs if not disabled
    if not args.no_display:
        display_outputs(outputs)
    
    # Add debug visualization functionality
    def debug_show_images(outputs, save_dir=None):
        """
        Display or save debug visualizations of shader outputs
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import os
        
        # Create save directory if needed
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Check if we're in a headless environment
        is_headless = not hasattr(plt, 'get_backend') or plt.get_backend() == 'agg'
        
        if is_headless:
            matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create figure with subplots for each output
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display color output
        if 'color' in outputs:
            axes[0].imshow(outputs['color'])
            axes[0].set_title('Color')
            axes[0].axis('off')
        
        # Display geometry
        if 'geometry' in outputs:
            # Normalize geometry to 0-1 range
            geometry_display = (outputs['geometry'] - outputs['geometry'].min()) / (outputs['geometry'].max() - outputs['geometry'].min() + 1e-8)
            axes[1].imshow(geometry_display)
            axes[1].set_title('Geometry')
            axes[1].axis('off')
        
        # Display normals (normalize to 0-1 range)
        if 'normals' in outputs:
            normals_display = (outputs['normals'] + 1) / 2  # normalize to 0-1
            axes[2].imshow(normals_display)
            axes[2].set_title('Normals')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if is_headless or save_dir:
            # Save to file
            save_path = os.path.join(save_dir or './debug/images', 'shader_debug.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved debug visualization to: {save_path}")
        else:
            # Display interactively
            plt.show()
    
    # Generate debug visualizations
    if args.save:
        debug_save_dir = os.path.join(args.output_dir, 'debug')
        debug_show_images(outputs, debug_save_dir)
    
    # Clean up
    state['ctx'].release()


if __name__ == "__main__":
    main()
