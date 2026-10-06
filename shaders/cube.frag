#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    layout(offset = 64) vec4 colorModifier;
} push;

void main() {
    outColor = vec4(fragColor, 1.0) * push.colorModifier;
}
