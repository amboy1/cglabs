#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;

// Set 0: Scene Uniforms
layout (std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    vec3 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

// Set 1, Binding 0: Dynamic Model Uniform
layout (std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec3 albedo;
    vec3 specular;
    float alpha;
} model_uniforms;

void main() {
    vec4 worldPos = model_uniforms.model * vec4(inPosition, 1.0);
    gl_Position = scene_uniforms.view_projection * worldPos;

    fragPosition = worldPos.xyz;
    mat3 normalMatrix = transpose(inverse(mat3(model_uniforms.model)));
    fragNormal = normalize(normalMatrix * inNormal);
    fragTexCoord = inTexCoord;
}