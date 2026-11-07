#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 angles;
};

// Set 0: Scene Uniforms & Light SSBOs
layout (std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    vec3 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

layout (std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout (std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

// Set 1: Dynamic Model Uniform + Текстурные карты (Albedo, Specular, Emissive)
layout (std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec3 albedo;
    vec3 specular;
    float alpha;
} model_uniforms;

layout (set = 1, binding = 1) uniform sampler2D texAlbedo;
layout (set = 1, binding = 2) uniform sampler2D texSpecular;
layout (set = 1, binding = 3) uniform sampler2D texEmissive;

// Push Constants для контроля нетривиального сэмплирования
layout(std140, push_constant) uniform PushConstants {
    int enable_special_sampling; // 1 = включено, 0 = обычное
    float time;
} push_consts;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene_uniforms.camera_position - fragPosition);

    // Выборка и модуляция текстурных координат
    vec3 albedoTexColor;
    if (push_consts.enable_special_sampling == 1) {
        // Нетривиальное параметризованное сэмплирование:
        // 1. Динамическое искажение координат по синусоиде
        float t = push_consts.time;
        vec2 uvWave = fragTexCoord + vec2(sin(fragTexCoord.y * 12.0 + t * 3.0) * 0.04, 
                                          cos(fragTexCoord.x * 12.0 + t * 3.0) * 0.04);
        
        // 2. Множественное сэмплирование (двумя сдвинутыми слоями с масштабированием)
        vec4 sample1 = texture(texAlbedo, uvWave);
        vec4 sample2 = texture(texAlbedo, uvWave * 2.0 + vec2(t * 0.1, -t * 0.1));
        
        // 3. Модуляция цвета (пульсирующий оттенок)
        vec3 tint = vec3(0.5 + 0.5 * sin(t), 0.7, 0.5 + 0.5 * cos(t));
        albedoTexColor = mix(sample1.rgb, sample2.rgb, 0.4) * tint;
    } else {
        albedoTexColor = texture(texAlbedo, fragTexCoord).rgb;
    }

    vec3 Kd = model_uniforms.albedo * albedoTexColor;
    // Specular map модулирует интенсивность и цвет зеркального отражения
    vec3 Ks = model_uniforms.specular * texture(texSpecular, fragTexCoord).rgb;
    float shininess = max(model_uniforms.alpha, 1.0);

    // 1. Ambient
    vec3 totalColor = scene_uniforms.ambient_light * Kd;

    // 2. Directional Light
    if (length(scene_uniforms.directional_light_color) > 0.0) {
        vec3 L = normalize(-scene_uniforms.directional_light_direction);
        float diff = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float spec = 0.0;
        if (diff > 0.0) {
            spec = pow(max(dot(N, H), 0.0), shininess);
        }
        totalColor += (diff * Kd + spec * Ks) * scene_uniforms.directional_light_color;
    }

    // 3. Point Lights
    for (uint i = 0u; i < scene_uniforms.point_light_count; ++i) {
        vec3 lightPos = point_lights[i].position.xyz;
        vec3 lightColor = point_lights[i].color.rgb * point_lights[i].color.w;

        vec3 toLight = lightPos - fragPosition;
        float dist = length(toLight);
        if (dist < 0.0001) continue;
        vec3 L = toLight / dist;

        float attenuation = 1.0 / max(dist * dist, 0.01);
        float diff = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float spec = 0.0;
        if (diff > 0.0) {
            spec = pow(max(dot(N, H), 0.0), shininess);
        }
        totalColor += (diff * Kd + spec * Ks) * lightColor * attenuation;
    }

    // 4. Spot Lights
    for (uint i = 0u; i < scene_uniforms.spot_light_count; ++i) {
        vec3 lightPos = spot_lights[i].position.xyz;
        vec3 lightDir = normalize(spot_lights[i].direction.xyz);
        vec3 lightColor = spot_lights[i].color.rgb * spot_lights[i].color.w;

        vec3 toLight = lightPos - fragPosition;
        float dist = length(toLight);
        if (dist < 0.0001) continue;
        vec3 L = toLight / dist;

        float cosTheta = dot(-L, lightDir);
        float cosInner = spot_lights[i].angles.x;
        float cosOuter = spot_lights[i].angles.y;

        float spotIntensity = clamp((cosTheta - cosOuter) / max(cosInner - cosOuter, 0.0001), 0.0, 1.0);
        if (spotIntensity <= 0.0) continue;

        float attenuation = 1.0 / max(dist * dist, 0.01);
        float diff = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float spec = 0.0;
        if (diff > 0.0) {
            spec = pow(max(dot(N, H), 0.0), shininess);
        }
        totalColor += (diff * Kd + spec * Ks) * lightColor * attenuation * spotIntensity;
    }

    // 5. Emissive (самосвечение без участия Blinn-Phong)
    vec3 emissive = texture(texEmissive, fragTexCoord).rgb;
    totalColor += emissive;

    outColor = vec4(totalColor, 1.0);
}