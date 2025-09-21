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
uniform vec3      iObjectOffset;

in vec2 v_texcoord;
in flat int v_object_id;

float intersection_count = 0.0;


#define ESCAPE_THRESHOLD 20.0
#define DEL 0.0001
#define BOUNDING_RADIUS_2 5

#define PI 3.14159265359

// out vec4 out_color;
// out vec3 out_coord;
// out vec3 out_normals;
// out vec3 out_debug;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_coord;
layout(location = 2) out vec3 out_normals;
layout(location = 3) out vec3 out_debug;
layout(location = 7) out vec2 out_uv;
layout(location = 4) out float out_depth;
layout(location = 5) out int out_object_id;
layout(location = 6) out float out_distance_field;

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
    // the (approximate) distance between the first point along the ray within
    // epsilon of some point in the Julia set, or the last point to be tested if
    // there was no intersection.
    float dist;
    
    int maxSteps = 200;
    float n;
    for (int i = 0; i < maxSteps; i++) {
        n = float(i);
        vec4 z = vec4(rO, 0);
        vec4 zp = vec4(1.0, vec3(0.0));
        // iterate this point until we can guess if the sequence diverges or converges.
        julia(z, zp, c, maxIterations);
        // find a lower bound on the distance to the Julia set and step this far along the ray.
        float normZ = length(z);
        dist = 0.5 * normZ * log(normZ) / length(zp); //lower bound on distance to surface
        // Spencer Note: Added the 0.8 to make sure we don't miss the surface. Could possibly make this dynamic.
        // based on the object offset. TODO: Make this dynamic. Below is the 0.8 scalar.
        rO += rD * (dist * 0.8); // (step) 
    
        // Intersection testing finishes if we’re close enough to the surface
        // (i.e., we’re inside the epsilon isosurface of the distance estimator
        // function) or have left the bounding sphere.
        if( dist < epsilon || dot(rO, rO) > BOUNDING_RADIUS_2)
            break;
    }
    // return the distance for this ray
    return dist;
}

vec3 blinn_phong(vec3 light, vec3 eye, vec3 pt, vec3 N) {
    vec3 baseColor = iMaterial;              // base color of shading
    if (iUseObjTexture) {
        vec3 pp = pt * 0.5 + 0.5;
        baseColor = texture(iObjTexture, pp).rgb;
    }

    vec3 L = normalize(light - pt);     // vector to the light
    vec3 E = -eye;                      // vector to the camera
    vec3 H = normalize((E + L) / 2.0);  // half vector

    float NdotL = dot(N, L);            // cosine of the angle between light and normal
    
    float ambiant = iAmbientLight;
    float diffuse = max(NdotL, 0.0) * iDiffuseScale;
    float spec = pow(max(dot(H, N), 0.0), iSpecularExp) * iSpecularScale;
    // compute the illumination using the Phong equation
    vec3 shade = baseColor * (ambiant + diffuse + spec);

    // gamma correction
    shade[0] = pow(shade[0], 1.0 / 2.2);
    shade[1] = pow(shade[1], 1.0 / 2.2);
    shade[2] = pow(shade[2], 1.0 / 2.2);

    return shade;
}

vec4 render(vec3 ro, vec3 rd, vec4 c, vec3 light, int maxIter, float epsilon) {
    const vec4 backgroundColor = vec4(0.0);
    vec4 color = backgroundColor;
    vec3 N = vec3(0.0);

    float dist = intersect(ro, rd, c, maxIter, epsilon);
    out_distance_field = dist;
    // We say that we found an intersection if our estimate of the distance to
    // the set is smaller than some small value epsilon. In this case we want
    // to do some shading / coloring.
    if(dist < epsilon) {
        out_coord += ro;
        intersection_count += 1.0;
        // Determine a "surface normal" which we’ll use for lighting calculations.
        N = normEstimate(ro, c, maxIter);
        color.xyz = blinn_phong(light, rd, ro, N);
        
        // If the shadow flag is on, determine if this point is in shadow
        if(true) {
            // The shadow ray will start at the intersection point and go
            // towards the point light. We initially move the ray origin
            // a little bit along the normal direction so that we don’t mistakenly
            // find an intersection with the same point again.
            vec3 L = normalize(light - ro);
            vec3 p = ro + N * epsilon * 2.;
            dist = intersect(p, L, c, maxIter, epsilon);
            if(dist < epsilon)
                // darken the shadowed areas, adding some smoothness using the normal
                color.xyz *= 0.8 + (dot(L, N) + 1.0) * 0.1;
        }
        color.w = 1.0;
    }
    else if (iUseBgTexture) {
        color.xyz = texture(iBgTexture, gl_FragCoord.xy / iResolution.xy).rgb;
        color.a = 1.0;
    }
    // Export the normal to the normal map
    out_normals += N;

    // Return the final color which is still the background color if we didn’t hit anything.
    return color;
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

vec3 convertLocation(vec2 offsets, float distance) {
    float theta = PI * offsets.x; // rotation around the y axis
    float phi = -PI * offsets.y;  // rotation around the x axis

    float cx = cos(phi), sx = sin(phi);
    float cy = cos(theta), sy = sin(theta);
    // combined rotation around y and x axes to get camera position
    // mat3 rot = mat3(
    //     cy, 0, -sy,
    //     sy * sx, cx, cy * sx,
    //     sy * cx, -sx, cy * cx);
    
    // vec3 location = rot * vec3(0.0, 0.0, dist);
    vec3 location = distance * vec3(sy * cx, -sx, cy * cx);
    return location;
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec3 res = iResolution;
    vec4 c = iJuliaC;

    // light
    vec3 lightSource = convertLocation(iLightSource.xy, 2 * iViewDistance);
    // camera
    vec3 camera = convertLocation(iViewAngleXY.xy, iViewDistance);
    camera += iObjectOffset;
	vec3 target = vec3(0.0, 0.0, 0.0);
    target += iObjectOffset;
    float roll = 0.0; // camera roll
    float focal = iFocalPlane; // focal plane (zoom)
    mat3 cameraMat = getCameraMatrix(camera, target, roll);
    out_coord = vec3(0.0);

    // antialiasing by averaging over multiple rays for a single pixel
    float mdim = max(res.x, res.y);
    int AA = max(1, iAntialias);
    float subpixel_size = AA;
    vec4 color;
    vec2 xy = (2.0 * fragCoord - res.xy) / mdim;
    for (int i = 0; i < AA; i++)
        for (int j = 0; j < AA; j++) {
            vec2 p = xy + vec2(float(i), float(j)) / (subpixel_size * mdim);
            vec3 rd = normalize(cameraMat * vec3(p, focal));
            color += render(camera, rd, c, lightSource, 10, 9e-3);
        }
    color /= float(AA * AA);
    // out_coord /= float(AA * AA);
    out_coord /= max(1.0, intersection_count);

    // out_normals = out_normals / float(AA * AA);
    float nlen = length(out_normals);
    // out_normals = nlen > 0 ? normalize(out_normals / float(AA * AA)) : vec3(0.0);
    out_normals = nlen > 0 ? normalize(out_normals / max(1.0, intersection_count)) : vec3(0.0);

    out_color = color;
    vec2 texcoord = v_texcoord;
    out_debug = vec3(texcoord, 1.0);
    
    // Output UV coordinates (normalized screen coordinates)
    out_uv = gl_FragCoord.xy / iResolution.xy;

    if (out_coord == vec3(0.0)) {
        out_depth = 200.0;
        out_object_id = -1;
        out_uv = vec2(0.0);  // Invalid UV for background
    }
    else {
        // Add back the x-offset to out_coord for correct depth calculation
        // out_coord += iObjectOffset;
        // vec3 adjusted_coord = out_coord;
        // adjusted_coord.x -= iObjectOffset.x;
        // adjusted_coord.y -= iObjectOffset.y;
        // adjusted_coord.z -= iObjectOffset.z;
        
        out_coord = out_coord - iObjectOffset;
        vec3 cameraToCoord = camera - out_coord;
        vec3 viewDir = normalize(camera - target);
        out_depth = dot(cameraToCoord, viewDir);
        out_object_id = v_object_id;
        // out_coord = out_coord - iObjectOffset;
        
        
    }
}



    
    
