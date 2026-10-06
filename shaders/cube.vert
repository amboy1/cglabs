#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(binding = 0) uniform ViewProjection {
    mat4 view;
    mat4 proj;
} vp;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 colorModifier;
} push;

void main() {
    gl_Position = vp.proj * vp.view * push.model * vec4(inPosition, 1.0);
    fragColor = inColor;
}
