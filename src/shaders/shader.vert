#version 320 es

layout (location = 0) in vec3 VertexPosition;

in vec4 v_color;
in vec3 v_normal;

uniform mat4 ModelViewMatrix;
uniform mat4 MVP;
uniform vec3 center;
uniform vec3 scale;
uniform bool scaleEnabled;

out vec3 Normal;
out vec3 Position;
out vec4 Color;

vec3 scaleNorm(in vec3 norm){
    norm = normalize(norm);
    float a = norm.x;
    float b = norm.y;
    float c = norm.z;
    
    vec3 p1 = vec3(0., 0., 0.); //0
    vec3 p2;
    vec3 p3;
    if (a != 0.) {
        p2 = vec3(-b / a, 1., 0.); //4
        p3 = vec3(-c / a, 0., 1.); //6
    }
    else if (c != 0.) {
        p2 = vec3(1., 0., -a / c); //5
        p3 = vec3(0., 1., -b / c); //1
    }
    else {
        p2 = vec3(0., -c / b, 1.); //2
        p3 = vec3(1., -a / b, 0.); //3
    }
    p2 = p2 * scale;
    p3 = p3 * scale;
    vec3 scaled_norm = normalize(cross(p2, p3));
    if (dot(norm, scaled_norm) < 0.) {
        scaled_norm = -scaled_norm;
    }
    return scaled_norm;
}

void main(){
    float k = 0.1;
    vec3 pos = vec3(VertexPosition);
    vec3 norm = vec3(v_normal);
    if (scaleEnabled && scale != vec3(1, 1, 1))  {
        pos = (pos - center) * scale + center;
        norm = scaleNorm(norm);
    }
    
    Position = pos;
    Color = v_color;
    Normal = norm;

    gl_Position = MVP * vec4(pos, 1.);
}
