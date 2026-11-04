#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position; // xyz: position, w: unused
    vec4 color;    // rgb: color, w: intensity
};

struct SpotLight {
    vec4 position;  // xyz: position, w: unused
    vec4 direction; // xyz: direction, w: unused
    vec4 color;     // rgb: color, w: intensity
    vec4 angles;    // x: cos(inner), y: cos(outer)
};

// Set 0: Scene-level uniforms and dynamic light storage buffers
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

// Set 1: Model-level uniform
layout (std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec3 albedo;
    vec3 specular;
    float alpha;
} model_uniforms;

void main() {
    vec3 N = normalize(fragNormal);
    // Вектор от точки на поверхности к камере
    vec3 V = normalize(scene_uniforms.camera_position - fragPosition);

    vec3 Kd = model_uniforms.albedo;
    vec3 Ks = model_uniforms.specular;
    float shininess = max(model_uniforms.alpha, 1.0);

    // 1. Рассеянный свет (Ambient)
    vec3 totalColor = scene_uniforms.ambient_light * Kd;

    // 2. Направленный источник света (Directional Light)
    if (length(scene_uniforms.directional_light_color) > 0.0) {
        // directional_light_direction указывает направление лучей света
        vec3 L = normalize(-scene_uniforms.directional_light_direction);
        float diff = max(dot(N, L), 0.0);

        // Blinn-Phong: полувектор H между направлением на свет и направлением на камеру
        vec3 H = normalize(L + V);
        float spec = 0.0;
        if (diff > 0.0) {
            spec = pow(max(dot(N, H), 0.0), shininess);
        }

        totalColor += (diff * Kd + spec * Ks) * scene_uniforms.directional_light_color;
    }

    // 3. Точечные источники света (Point Lights) со спадом по закону обратных квадратов
    for (uint i = 0u; i < scene_uniforms.point_light_count; ++i) {
        vec3 lightPos = point_lights[i].position.xyz;
        vec3 lightColor = point_lights[i].color.rgb * point_lights[i].color.w; // цвет * интенсивность

        vec3 toLight = lightPos - fragPosition;
        float dist = length(toLight);
        if (dist < 0.0001) continue;
        vec3 L = toLight / dist;

        // Затухание: 1 / D^2 (с защитой от деления на ноль)
        float attenuation = 1.0 / max(dist * dist, 0.01);

        float diff = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float spec = 0.0;
        if (diff > 0.0) {
            spec = pow(max(dot(N, H), 0.0), shininess);
        }

        totalColor += (diff * Kd + spec * Ks) * lightColor * attenuation; //Блинн - Фонг
    }

    // 4. Прожекторные источники света (Spot Lights) со спадом 1 / D^2 и мягкими краями
    for (uint i = 0u; i < scene_uniforms.spot_light_count; ++i) {
        vec3 lightPos = spot_lights[i].position.xyz;
        vec3 lightDir = normalize(spot_lights[i].direction.xyz);
        vec3 lightColor = spot_lights[i].color.rgb * spot_lights[i].color.w; // цвет * интенсивность

        vec3 toLight = lightPos - fragPosition;
        float dist = length(toLight);
        if (dist < 0.0001) continue;
        vec3 L = toLight / dist;

        // Вектор луча прожектора от источника к фрагменту: -L
        // Угол отклонения фрагмента от оси прожектора
        float cosTheta = dot(-L, lightDir);
        float cosInner = spot_lights[i].angles.x;
        float cosOuter = spot_lights[i].angles.y;

        // Мягкие края: спад интенсивности от внутреннего угла к внешнему
        float spotIntensity = clamp((cosTheta - cosOuter) / max(cosInner - cosOuter, 0.0001), 0.0, 1.0);
        if (spotIntensity <= 0.0) continue;

        // Затухание по закону обратных квадратов
        float attenuation = 1.0 / max(dist * dist, 0.01);

        float diff = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float spec = 0.0;
        if (diff > 0.0) {
            spec = pow(max(dot(N, H), 0.0), shininess);
        }

        totalColor += (diff * Kd + spec * Ks) * lightColor * attenuation * spotIntensity;
    }

    outColor = vec4(totalColor, 1.0);
}
