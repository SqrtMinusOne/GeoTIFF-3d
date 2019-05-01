#version 320 es

layout (location = 0) in vec3 VertexPosition;

in vec4 v_color;
in vec3 v_normal;

uniform mat4 ModelViewMatrix;
uniform mat4 MVP;

out vec3 Normal;
out vec3 Position;
out vec4 Color;

void main(){
    float k = 0.1;
    vec4 pos = vec4(VertexPosition, 1.); 
    Position = VertexPosition;
    
    Color = v_color;
    
    Normal = v_normal;

    gl_Position = MVP * pos;

}
