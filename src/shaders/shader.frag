#version 320 es
#undef lowp
#undef mediump
#undef highp

precision mediump float;

uniform vec3 LightPos;
uniform float ambientStrength;
uniform float diffuseStrength;
uniform float alpha;
uniform bool phongModel;

in vec4 Color;
in vec3 Normal;
in vec3 Position;

out vec4 fragColor;

void main() {
    vec4 lightColor = vec4(1.0, 1.0, 1.0, 1.0);
    
    if (phongModel){
        vec4 ambient = ambientStrength * lightColor;
        
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(LightPos - Position);
        float diff = max(dot(norm, lightDir), 0.0);
        vec4 diffuse = diff * diffuseStrength * lightColor;

        vec4 result = (ambient + diffuse) * Color;
        result.a = alpha;

        fragColor = result;
    }
    else {
        fragColor = Color;
    }
}

