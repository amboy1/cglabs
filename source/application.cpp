#include "application.hpp"

#include <imgui.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {

namespace {

// =========================================================================
// Геометрия и процедурная генерация нормалей
// =========================================================================
struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec3 color;
};

std::vector<Vertex> vertices;
std::vector<uint16_t> indices;

void generateCube(std::vector<Vertex> &outVertices,
                  std::vector<uint16_t> &outIndices) {
  outVertices.clear();
  outIndices.clear();

  struct Face {
    glm::vec3 p0, p1, p2, p3;
    glm::vec3 color;
  };

  // 6 граней куба (вершины ориентированы CCW при взгляде снаружи)
  const std::array<Face, 6> faces = {{// Передняя грань (+Z)
                                      {{-0.5f, -0.5f, 0.5f},
                                       {0.5f, -0.5f, 0.5f},
                                       {0.5f, 0.5f, 0.5f},
                                       {-0.5f, 0.5f, 0.5f},
                                       {0.9f, 0.2f, 0.2f}},
                                      // Задняя грань (-Z)
                                      {{0.5f, -0.5f, -0.5f},
                                       {-0.5f, -0.5f, -0.5f},
                                       {-0.5f, 0.5f, -0.5f},
                                       {0.5f, 0.5f, -0.5f},
                                       {0.2f, 0.8f, 0.2f}},
                                      // Верхняя грань (+Y)
                                      {{-0.5f, 0.5f, 0.5f},
                                       {0.5f, 0.5f, 0.5f},
                                       {0.5f, 0.5f, -0.5f},
                                       {-0.5f, 0.5f, -0.5f},
                                       {0.2f, 0.4f, 0.9f}},
                                      // Нижняя грань (-Y)
                                      {{-0.5f, -0.5f, -0.5f},
                                       {0.5f, -0.5f, -0.5f},
                                       {0.5f, -0.5f, 0.5f},
                                       {-0.5f, -0.5f, 0.5f},
                                       {0.9f, 0.9f, 0.2f}},
                                      // Правая грань (+X)
                                      {{0.5f, -0.5f, 0.5f},
                                       {0.5f, -0.5f, -0.5f},
                                       {0.5f, 0.5f, -0.5f},
                                       {0.5f, 0.5f, 0.5f},
                                       {0.9f, 0.2f, 0.9f}},
                                      // Левая грань (-X)
                                      {{-0.5f, -0.5f, -0.5f},
                                       {-0.5f, -0.5f, 0.5f},
                                       {-0.5f, 0.5f, 0.5f},
                                       {-0.5f, 0.5f, -0.5f},
                                       {0.2f, 0.9f, 0.9f}}}};

  for (const auto &face : faces) {
    auto startIndex = static_cast<uint16_t>(outVertices.size());

    // 4 вершины на грань
    outVertices.push_back({face.p0, glm::vec3(0.0f), face.color});
    outVertices.push_back({face.p1, glm::vec3(0.0f), face.color});
    outVertices.push_back({face.p2, glm::vec3(0.0f), face.color});
    outVertices.push_back({face.p3, glm::vec3(0.0f), face.color});

    // 2 треугольника на грань (0, 1, 2) и (2, 3, 0)
    outIndices.push_back(startIndex + 0);
    outIndices.push_back(startIndex + 1);
    outIndices.push_back(startIndex + 2);

    outIndices.push_back(startIndex + 2);
    outIndices.push_back(startIndex + 3);
    outIndices.push_back(startIndex + 0);
  }
}

// Процедурный расчет нормалей вершин по треугольникам
void generateProceduralNormals(std::vector<Vertex> &verts,
                               const std::vector<uint16_t> &inds) {
  for (auto &v : verts) {
    v.normal = glm::vec3(0.0f);
  }

  for (size_t i = 0; i < inds.size(); i += 3) {
    uint16_t i0 = inds[i];
    uint16_t i1 = inds[i + 1];
    uint16_t i2 = inds[i + 2];

    glm::vec3 v0 = verts[i0].pos;
    glm::vec3 v1 = verts[i1].pos;
    glm::vec3 v2 = verts[i2].pos;

    // Векторное произведение ребер дает нормаль к плоскости треугольника
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 faceNormal = glm::cross(edge1, edge2);

    verts[i0].normal += faceNormal;
    verts[i1].normal += faceNormal;
    verts[i2].normal += faceNormal;
  }

  // Нормализуем полученные нормали
  for (auto &v : verts) {
    if (glm::length(v.normal) > 0.0001f) {
      v.normal = glm::normalize(v.normal);
    }
  }
}

// Set = 0, Binding = 0: SceneUniforms (std140)
struct SceneUniformsGPU {
  glm::mat4 view_projection;
  alignas(16) glm::vec3 camera_position;
  alignas(16) glm::vec3 directional_light_direction;
  alignas(16) glm::vec3 directional_light_color;
  alignas(16) glm::vec3 ambient_light;
  uint32_t point_light_count;
  uint32_t spot_light_count;
  float _pad[2];
};
static_assert(sizeof(SceneUniformsGPU) == 144,
              "SceneUniformsGPU layout mismatch");

// Set = 1, Binding = 0: ModelUniform (std140)
struct ModelUniformGPU {
  glm::mat4 model;
  alignas(16) glm::vec3 albedo;
  alignas(16) glm::vec3 specular;
  float alpha;
};
static_assert(sizeof(ModelUniformGPU) == 96, "ModelUniformGPU layout mismatch");

// Set = 0, Binding = 1: PointLight (std430 SSBO)
struct PointLightGPU {
  glm::vec4 position; // xyz = pos, w = pad
  glm::vec4 color;    // rgb = color, w = intensity
};
static_assert(sizeof(PointLightGPU) == 32, "PointLightGPU layout mismatch");

// Set = 0, Binding = 2: SpotLight (std430 SSBO)
struct SpotLightGPU {
  glm::vec4 position;  // xyz = pos, w = pad
  glm::vec4 direction; // xyz = dir, w = pad
  glm::vec4 color;     // rgb = color, w = intensity
  glm::vec4
      angles; // x = cos(inner), y = cos(outer), z = innerDeg, w = outerDeg
};
static_assert(sizeof(SpotLightGPU) == 64, "SpotLightGPU layout mismatch");

constexpr uint32_t MAX_LIGHTS = 32;

// =========================================================================
// UI Структуры
// =========================================================================
struct PointLightUI {
  glm::vec3 position{1.2f, 1.2f, 1.0f};
  glm::vec3 color{1.0f, 0.7f, 0.3f};
  float intensity{2.5f};
};

struct SpotLightUI {
  glm::vec3 position{-1.5f, 2.0f, 1.5f};
  glm::vec3 direction{1.5f, -2.0f, -1.5f};
  glm::vec3 color{0.3f, 0.7f, 1.0f};
  float intensity{6.0f};
  float innerAngle{15.0f};
  float outerAngle{28.0f};
};

// =========================================================================
// Ресурсы Vulkan
// =========================================================================
VkBuffer vertexBuffer;
VmaAllocation vertexBufferAllocation;

VkBuffer indexBuffer;
VmaAllocation indexBufferAllocation;

// Буферы Uniform и Storage
VkBuffer sceneUniformBuffer;
VmaAllocation sceneUniformBufferAllocation;
void *sceneUniformBufferMapped = nullptr;

VkBuffer modelUniformBuffer;
VmaAllocation modelUniformBufferAllocation;
void *modelUniformBufferMapped = nullptr;

VkBuffer pointLightsBuffer;
VmaAllocation pointLightsBufferAllocation;
void *pointLightsBufferMapped = nullptr;

VkBuffer spotLightsBuffer;
VmaAllocation spotLightsBufferAllocation;
void *spotLightsBufferMapped = nullptr;

// Два набора дескрипторов (Set 0: сцена, Set 1: модель)
VkDescriptorSetLayout sceneDescriptorSetLayout;
VkDescriptorSetLayout modelDescriptorSetLayout;
VkDescriptorPool descriptorPool;
VkDescriptorSet sceneDescriptorSet;
VkDescriptorSet modelDescriptorSet;

VkPipelineLayout pipelineLayout;
VkPipeline graphicsPipeline;

// =========================================================================
// Состояние сцены и UI
// =========================================================================

// Камера (позиция, поворот, FOV)
glm::vec3 cameraPos = glm::vec3(0.0f, 1.5f, 3.5f);
float cameraYaw = -90.0f;
float cameraPitch = -18.0f;
float fov = 45.0f;
bool usePerspective = true;
float orthoSize = 2.0f;

// Материал объекта (по модели Блинн-Фонга)
glm::vec3 materialAlbedo = glm::vec3(0.85f, 0.25f, 0.25f);
glm::vec3 materialSpecular = glm::vec3(1.0f, 1.0f, 1.0f);
float materialShininess = 64.0f;

// Трансформация объекта
glm::vec3 cubePosition = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cubeRotation = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cubeScale = glm::vec3(1.0f, 1.0f, 1.0f);

// Анимация
bool animate = false;
float animSpeed = 1.0f;
float currentAnimTime = 0.0f;

// Окружающий свет (Ambient light)
glm::vec3 ambientLightColor = glm::vec3(0.12f, 0.12f, 0.15f);

// Направленный свет (Directional light)
bool dirLightEnabled = true;
glm::vec3 dirLightDirection = glm::vec3(-0.6f, -1.0f, -0.5f);
glm::vec3 dirLightColor = glm::vec3(1.0f, 0.95f, 0.85f);
float dirLightIntensity = 1.0f;

// Массивы точечных и прожекторных источников
std::vector<PointLightUI> pointLights = {
    {glm::vec3(1.3f, 1.2f, 1.0f), glm::vec3(1.0f, 0.6f, 0.2f), 2.5f}};

std::vector<SpotLightUI> spotLights = {
    {glm::vec3(-1.5f, 1.8f, 1.5f), glm::vec3(1.5f, -1.8f, -1.5f),
     glm::vec3(0.3f, 0.7f, 1.0f), 7.0f, 15.0f, 28.0f}};

std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file: " + filename);
  }
  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();
  return buffer;
}

VkShaderModule createShaderModule(const std::vector<char> &code) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(graphics::internal::context.device, &createInfo,
                           nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }
  return shaderModule;
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                  VmaMemoryUsage memoryUsage, VkBuffer &buffer,
                  VmaAllocation &allocation) {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = memoryUsage;
  allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

  if (vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo,
                      &allocInfo, &buffer, &allocation,
                      nullptr) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }
}

} // namespace

bool initialize() {
  auto &ctx = graphics::internal::context;

  // 1. Процедурная генерация куба и нормалей
  generateCube(vertices, indices);
  generateProceduralNormals(vertices, indices);

  // 2. Vertex buffer
  VkDeviceSize vertexBufferSize = sizeof(Vertex) * vertices.size();
  createBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
               VMA_MEMORY_USAGE_AUTO, vertexBuffer, vertexBufferAllocation);

  void *data = nullptr;
  vmaMapMemory(ctx.allocator, vertexBufferAllocation, &data);
  memcpy(data, vertices.data(), (size_t)vertexBufferSize);
  vmaUnmapMemory(ctx.allocator, vertexBufferAllocation);

  // 3. Index buffer
  VkDeviceSize indexBufferSize = sizeof(indices[0]) * indices.size();
  createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
               VMA_MEMORY_USAGE_AUTO, indexBuffer, indexBufferAllocation);

  vmaMapMemory(ctx.allocator, indexBufferAllocation, &data);
  memcpy(data, indices.data(), (size_t)indexBufferSize);
  vmaUnmapMemory(ctx.allocator, indexBufferAllocation);

  // 4. Uniform буфер сцены (Set 0, Binding 0)
  createBuffer(sizeof(SceneUniformsGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
               VMA_MEMORY_USAGE_AUTO, sceneUniformBuffer,
               sceneUniformBufferAllocation);
  vmaMapMemory(ctx.allocator, sceneUniformBufferAllocation,
               &sceneUniformBufferMapped);

  // 5. Storage буфер точечных источников (Set 0, Binding 1)
  createBuffer(sizeof(PointLightGPU) * MAX_LIGHTS,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
               pointLightsBuffer, pointLightsBufferAllocation);
  vmaMapMemory(ctx.allocator, pointLightsBufferAllocation,
               &pointLightsBufferMapped);

  // 6. Storage буфер прожекторов (Set 0, Binding 2)
  createBuffer(sizeof(SpotLightGPU) * MAX_LIGHTS,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
               spotLightsBuffer, spotLightsBufferAllocation);
  vmaMapMemory(ctx.allocator, spotLightsBufferAllocation,
               &spotLightsBufferMapped);

  // 7. Uniform буфер модели (Set 1, Binding 0)
  createBuffer(sizeof(ModelUniformGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
               VMA_MEMORY_USAGE_AUTO, modelUniformBuffer,
               modelUniformBufferAllocation);
  vmaMapMemory(ctx.allocator, modelUniformBufferAllocation,
               &modelUniformBufferMapped);

  // =========================================================================
  // Создание Descriptor Set Layouts (Set 0 и Set 1)
  // =========================================================================

  // Set 0 Layout: Scene (UBO + 2 SSBO)
  std::array<VkDescriptorSetLayoutBinding, 3> sceneBindings{};
  // binding 0: SceneUniforms
  sceneBindings[0].binding = 0;
  sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sceneBindings[0].descriptorCount = 1;
  sceneBindings[0].stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  // binding 1: PointLights (Storage Buffer)
  sceneBindings[1].binding = 1;
  sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sceneBindings[1].descriptorCount = 1;
  sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  // binding 2: SpotLights (Storage Buffer)
  sceneBindings[2].binding = 2;
  sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sceneBindings[2].descriptorCount = 1;
  sceneBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{};
  sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  sceneLayoutInfo.bindingCount = static_cast<uint32_t>(sceneBindings.size());
  sceneLayoutInfo.pBindings = sceneBindings.data();

  if (vkCreateDescriptorSetLayout(ctx.device, &sceneLayoutInfo, nullptr,
                                  &sceneDescriptorSetLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create scene descriptor set layout!");
  }

  // Set 1 Layout: Model (UBO)
  VkDescriptorSetLayoutBinding modelBinding{};
  modelBinding.binding = 0;
  modelBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  modelBinding.descriptorCount = 1;
  modelBinding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo modelLayoutInfo{};
  modelLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  modelLayoutInfo.bindingCount = 1;
  modelLayoutInfo.pBindings = &modelBinding;

  if (vkCreateDescriptorSetLayout(ctx.device, &modelLayoutInfo, nullptr,
                                  &modelDescriptorSetLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create model descriptor set layout!");
  }

  // =========================================================================
  // Descriptor Pool & Allocation
  // =========================================================================
  std::array<VkDescriptorPoolSize, 2> poolSizes{};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = 2; // scene UBO + model UBO
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[1].descriptorCount = 2; // point lights SSBO + spot lights SSBO

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = 2; // Set 0 + Set 1

  if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }

  // Выделяем Set 0 и Set 1
  VkDescriptorSetAllocateInfo allocInfoScene{};
  allocInfoScene.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfoScene.descriptorPool = descriptorPool;
  allocInfoScene.descriptorSetCount = 1;
  allocInfoScene.pSetLayouts = &sceneDescriptorSetLayout;

  if (vkAllocateDescriptorSets(ctx.device, &allocInfoScene,
                               &sceneDescriptorSet) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate scene descriptor set!");
  }

  VkDescriptorSetAllocateInfo allocInfoModel{};
  allocInfoModel.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfoModel.descriptorPool = descriptorPool;
  allocInfoModel.descriptorSetCount = 1;
  allocInfoModel.pSetLayouts = &modelDescriptorSetLayout;

  if (vkAllocateDescriptorSets(ctx.device, &allocInfoModel,
                               &modelDescriptorSet) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate model descriptor set!");
  }

  // Обновляем Set 0
  VkDescriptorBufferInfo sceneUboInfo{};
  sceneUboInfo.buffer = sceneUniformBuffer;
  sceneUboInfo.offset = 0;
  sceneUboInfo.range = sizeof(SceneUniformsGPU);

  VkDescriptorBufferInfo pointSsboInfo{};
  pointSsboInfo.buffer = pointLightsBuffer;
  pointSsboInfo.offset = 0;
  pointSsboInfo.range = sizeof(PointLightGPU) * MAX_LIGHTS;

  VkDescriptorBufferInfo spotSsboInfo{};
  spotSsboInfo.buffer = spotLightsBuffer;
  spotSsboInfo.offset = 0;
  spotSsboInfo.range = sizeof(SpotLightGPU) * MAX_LIGHTS;

  std::array<VkWriteDescriptorSet, 3> sceneWrites{};
  sceneWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  sceneWrites[0].dstSet = sceneDescriptorSet;
  sceneWrites[0].dstBinding = 0;
  sceneWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sceneWrites[0].descriptorCount = 1;
  sceneWrites[0].pBufferInfo = &sceneUboInfo;

  sceneWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  sceneWrites[1].dstSet = sceneDescriptorSet;
  sceneWrites[1].dstBinding = 1;
  sceneWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sceneWrites[1].descriptorCount = 1;
  sceneWrites[1].pBufferInfo = &pointSsboInfo;

  sceneWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  sceneWrites[2].dstSet = sceneDescriptorSet;
  sceneWrites[2].dstBinding = 2;
  sceneWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sceneWrites[2].descriptorCount = 1;
  sceneWrites[2].pBufferInfo = &spotSsboInfo;

  vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(sceneWrites.size()),
                         sceneWrites.data(), 0, nullptr);

  // Обновляем Set 1
  VkDescriptorBufferInfo modelUboInfo{};
  modelUboInfo.buffer = modelUniformBuffer;
  modelUboInfo.offset = 0;
  modelUboInfo.range = sizeof(ModelUniformGPU);

  VkWriteDescriptorSet modelWrite{};
  modelWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  modelWrite.dstSet = modelDescriptorSet;
  modelWrite.dstBinding = 0;
  modelWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  modelWrite.descriptorCount = 1;
  modelWrite.pBufferInfo = &modelUboInfo;

  vkUpdateDescriptorSets(ctx.device, 1, &modelWrite, 0, nullptr);

  // =========================================================================
  // Pipeline Layout с двумя наборами дескрипторов (Set 0 и Set 1)
  // =========================================================================
  std::array<VkDescriptorSetLayout, 2> setLayouts = {sceneDescriptorSetLayout,
                                                     modelDescriptorSetLayout};

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
  pipelineLayoutInfo.pSetLayouts = setLayouts.data();
  pipelineLayoutInfo.pushConstantRangeCount = 0;
  pipelineLayoutInfo.pPushConstantRanges = nullptr;

  if (vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr,
                             &pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  // =========================================================================
  // Graphics Pipeline
  // =========================================================================
  auto vertShaderCode = readFile("shaders/cube.vert.spv");
  auto fragShaderCode = readFile("shaders/cube.frag.spv");

  VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
  VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

  VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
  vertShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
  fragShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                    fragShaderStageInfo};

  VkVertexInputBindingDescription bindingDescription{};
  bindingDescription.binding = 0;
  bindingDescription.stride = sizeof(Vertex);
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::vector<VkVertexInputAttributeDescription> attributeDescriptions(3);
  // Location 0: Позиция
  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, pos);

  // Location 1: Нормаль (сгенерированная процедурно)
  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, normal);

  // Location 2: Цвет
  attributeDescriptions[2].binding = 0;
  attributeDescriptions[2].location = 2;
  attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[2].offset = offsetof(Vertex, color);

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attributeDescriptions.size());
  vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;

  std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.renderPass = ctx.render_pass;
  pipelineInfo.subpass = 0;

  if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                nullptr, &graphicsPipeline) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }

  vkDestroyShaderModule(ctx.device, fragShaderModule, nullptr);
  vkDestroyShaderModule(ctx.device, vertShaderModule, nullptr);

  return true;
}

void shutdown() {
  auto &ctx = graphics::internal::context;
  vkQueueWaitIdle(ctx.graphics_queue);

  vkDestroyPipeline(ctx.device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);

  vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(ctx.device, modelDescriptorSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(ctx.device, sceneDescriptorSetLayout, nullptr);

  vmaUnmapMemory(ctx.allocator, modelUniformBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, modelUniformBuffer,
                   modelUniformBufferAllocation);

  vmaUnmapMemory(ctx.allocator, spotLightsBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, spotLightsBuffer, spotLightsBufferAllocation);

  vmaUnmapMemory(ctx.allocator, pointLightsBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, pointLightsBuffer,
                   pointLightsBufferAllocation);

  vmaUnmapMemory(ctx.allocator, sceneUniformBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, sceneUniformBuffer,
                   sceneUniformBufferAllocation);

  vmaDestroyBuffer(ctx.allocator, indexBuffer, indexBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, vertexBuffer, vertexBufferAllocation);
}

void update(double time) {
  static double lastTime = time;
  double dt = time - lastTime;
  lastTime = time;

  ImGui::SetNextWindowSize(ImVec2(440, 680), ImGuiCond_FirstUseEver);
  ImGui::Begin("Lab #2: Camera and Lighting");

  // =======================================================================
  // 1. Camera (position, rotation, FOV)
  // =======================================================================
  if (ImGui::CollapsingHeader("1. Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Projection");
    ImGui::RadioButton("Perspective", (int *)&usePerspective, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Orthographic", (int *)&usePerspective, 0);

    if (usePerspective) {
      ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f, "%.1f deg");
    } else {
      ImGui::SliderFloat("Ortho Size", &orthoSize, 0.5f, 10.0f, "%.2f");
    }

    ImGui::Separator();
    ImGui::DragFloat3("Position", glm::value_ptr(cameraPos), 0.05f);
    ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Pitch", &cameraPitch, -89.0f, 89.0f, "%.1f deg");

    if (ImGui::Button("Reset Camera")) {
      cameraPos = glm::vec3(0.0f, 1.5f, 3.5f);
      cameraYaw = -90.0f;
      cameraPitch = -18.0f;
      fov = 45.0f;
    }
  }

  // =======================================================================
  // 2. Object & Material (Blinn-Phong)
  // =======================================================================
  if (ImGui::CollapsingHeader("2. Object & Material (Blinn-Phong)",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::ColorEdit3("Albedo (Base Color)", glm::value_ptr(materialAlbedo));
    ImGui::ColorEdit3("Specular Color", glm::value_ptr(materialSpecular));
    ImGui::SliderFloat("Shininess (Alpha)", &materialShininess, 1.0f, 256.0f,
                       "%.1f");

    ImGui::Separator();
    ImGui::Text("Cube Transform");
    ImGui::DragFloat3("Position", glm::value_ptr(cubePosition), 0.05f);
    ImGui::DragFloat3("Rotation", glm::value_ptr(cubeRotation), 1.0f, -180.0f,
                      180.0f);
    ImGui::DragFloat3("Scale", glm::value_ptr(cubeScale), 0.02f, 0.1f, 5.0f);

    ImGui::Checkbox("Animate Cube", &animate);
    if (animate) {
      ImGui::SliderFloat("Anim Speed", &animSpeed, 0.1f, 5.0f);
    }
  }

  // =======================================================================
  // 3. Global Light (Ambient & Directional)
  // =======================================================================
  if (ImGui::CollapsingHeader("3. Global Light (Ambient & Directional)",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::ColorEdit3("Ambient Light", glm::value_ptr(ambientLightColor));

    ImGui::Separator();
    ImGui::Checkbox("Enable Directional Light", &dirLightEnabled);
    if (dirLightEnabled) {
      ImGui::SliderFloat3("Direction", glm::value_ptr(dirLightDirection), -1.0f,
                          1.0f);
      ImGui::ColorEdit3("Color", glm::value_ptr(dirLightColor));
      ImGui::SliderFloat("Intensity", &dirLightIntensity, 0.0f, 5.0f, "%.2f");
    }
  }

  // =======================================================================
  // 4. Point Lights (SSBO)
  // =======================================================================
  if (ImGui::CollapsingHeader("4. Point Lights (SSBO)",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Falloff: Inverse Square Law (1 / D^2)");
    if (pointLights.size() < MAX_LIGHTS) {
      if (ImGui::Button("+ Add Point Light")) {
        pointLights.push_back(
            {glm::vec3(0.0f, 1.5f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 2.0f});
      }
    } else {
      ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Max lights reached (%u)",
                         MAX_LIGHTS);
    }

    int toRemove = -1;
    for (size_t i = 0; i < pointLights.size(); ++i) {
      ImGui::PushID(static_cast<int>(i + 100));
      std::string headerName = "Point Light #" + std::to_string(i + 1);
      if (ImGui::TreeNode(headerName.c_str())) {
        ImGui::DragFloat3("Position", glm::value_ptr(pointLights[i].position),
                          0.05f);
        ImGui::ColorEdit3("Color", glm::value_ptr(pointLights[i].color));
        ImGui::SliderFloat("Intensity", &pointLights[i].intensity, 0.0f, 20.0f,
                           "%.2f");

        if (ImGui::Button("Delete Light")) {
          toRemove = static_cast<int>(i);
        }
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
    if (toRemove >= 0) {
      pointLights.erase(pointLights.begin() + toRemove);
    }
  }

  // =======================================================================
  // 5. Spot Lights (SSBO)
  // =======================================================================
  if (ImGui::CollapsingHeader("5. Spot Lights (SSBO)",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Falloff: 1 / D^2 and Soft Edges");
    if (spotLights.size() < MAX_LIGHTS) {
      if (ImGui::Button("+ Add Spot Light")) {
        spotLights.push_back({glm::vec3(0.0f, 2.0f, 1.5f),
                              glm::vec3(0.0f, -1.0f, -1.0f),
                              glm::vec3(1.0f, 1.0f, 0.8f), 5.0f, 15.0f, 25.0f});
      }
    } else {
      ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Max spot lights reached (%u)",
                         MAX_LIGHTS);
    }

    int toRemoveSpot = -1;
    for (size_t i = 0; i < spotLights.size(); ++i) {
      ImGui::PushID(static_cast<int>(i + 200));
      std::string headerName = "Spot Light #" + std::to_string(i + 1);
      if (ImGui::TreeNode(headerName.c_str())) {
        ImGui::DragFloat3("Position", glm::value_ptr(spotLights[i].position),
                          0.05f);
        ImGui::SliderFloat3(
            "Direction", glm::value_ptr(spotLights[i].direction), -1.0f, 1.0f);
        ImGui::ColorEdit3("Color", glm::value_ptr(spotLights[i].color));
        ImGui::SliderFloat("Intensity", &spotLights[i].intensity, 0.0f, 30.0f,
                           "%.2f");

        ImGui::SliderFloat("Inner Angle", &spotLights[i].innerAngle, 1.0f,
                           60.0f, "%.1f deg");
        ImGui::SliderFloat("Outer Angle (Soft Edge)", &spotLights[i].outerAngle,
                           2.0f, 80.0f, "%.1f deg");
        if (spotLights[i].outerAngle < spotLights[i].innerAngle) {
          spotLights[i].outerAngle = spotLights[i].innerAngle;
        }

        if (ImGui::Button("Delete Spot Light")) {
          toRemoveSpot = static_cast<int>(i);
        }
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
    if (toRemoveSpot >= 0) {
      spotLights.erase(spotLights.begin() + toRemoveSpot);
    }
  }

  ImGui::End();

  // Обновление анимации
  if (animate) {
    currentAnimTime += static_cast<float>(dt) * animSpeed;
    cubeRotation.y = currentAnimTime * 45.0f;
    cubeRotation.x = sin(currentAnimTime * 1.5f) * 15.0f;
  }
}

void render(const graphics::internal::FrameData &fd) {
  vkResetCommandBuffer(fd.command_buffer, 0);

  VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(fd.command_buffer, &begin_info);

  VkClearValue clear_values[2] = {};
  clear_values[0].color = {{0.05f, 0.05f, 0.07f, 1.0f}};
  clear_values[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = graphics::internal::context.render_pass,
      .framebuffer = fd.framebuffer,
      .renderArea =
          {
              .offset = {0, 0},
              .extent = graphics::internal::context.swapchain_extent,
          },
      .clearValueCount = 2,
      .pClearValues = clear_values,
  };

  vkCmdBeginRenderPass(fd.command_buffer, &render_pass_info,
                       VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphicsPipeline);

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width =
      static_cast<float>(graphics::internal::context.swapchain_extent.width);
  viewport.height =
      static_cast<float>(graphics::internal::context.swapchain_extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = graphics::internal::context.swapchain_extent;
  vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

  // =========================================================================
  // Вычисление матриц Камеры (View и Projection)
  // =========================================================================
  glm::vec3 front;
  front.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
  front.y = sin(glm::radians(cameraPitch));
  front.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
  glm::vec3 cameraFront = glm::normalize(front);
  glm::vec3 cameraRight =
      glm::normalize(glm::cross(cameraFront, glm::vec3(0.0f, 1.0f, 0.0f)));
  glm::vec3 cameraUp = glm::normalize(glm::cross(cameraRight, cameraFront));

  glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront,
                               cameraUp); // lookAt вычисляет обратную матрицу

  float aspect =
      viewport.width / (viewport.height > 0.0f ? viewport.height : 1.0f);
  glm::mat4 proj;
  if (usePerspective) {
    proj = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.0f);
  } else {
    float orthoHeight = orthoSize;
    float orthoWidth = orthoHeight * aspect;
    proj = glm::ortho(-orthoWidth, orthoWidth, -orthoHeight, orthoHeight,
                      -10.0f, 100.0f);
  }
  proj[1][1] *= -1; // Vulkan Y-flip

  // =========================================================================
  // Заполнение SceneUniforms (Set 0, Binding 0)
  // =========================================================================
  SceneUniformsGPU sceneGPU{};
  sceneGPU.view_projection = proj * view;
  sceneGPU.camera_position = cameraPos;
  sceneGPU.directional_light_direction = glm::normalize(dirLightDirection);
  sceneGPU.directional_light_color =
      dirLightEnabled ? (dirLightColor * dirLightIntensity) : glm::vec3(0.0f);
  sceneGPU.ambient_light = ambientLightColor;
  sceneGPU.point_light_count = static_cast<uint32_t>(
      std::min(pointLights.size(), static_cast<size_t>(MAX_LIGHTS)));
  sceneGPU.spot_light_count = static_cast<uint32_t>(
      std::min(spotLights.size(), static_cast<size_t>(MAX_LIGHTS)));

  memcpy(sceneUniformBufferMapped, &sceneGPU, sizeof(sceneGPU));

  // =========================================================================
  // Заполнение SSBO точечных источников (Set 0, Binding 1)
  // =========================================================================
  if (sceneGPU.point_light_count > 0) {
    std::vector<PointLightGPU> pointGpuData(sceneGPU.point_light_count);
    for (size_t i = 0; i < sceneGPU.point_light_count; ++i) {
      pointGpuData[i].position = glm::vec4(pointLights[i].position, 1.0f);
      pointGpuData[i].color =
          glm::vec4(pointLights[i].color, pointLights[i].intensity);
    }
    memcpy(pointLightsBufferMapped, pointGpuData.data(),
           sizeof(PointLightGPU) * pointGpuData.size());
  }

  // =========================================================================
  // Заполнение SSBO прожекторов (Set 0, Binding 2)
  // =========================================================================
  if (sceneGPU.spot_light_count > 0) {
    std::vector<SpotLightGPU> spotGpuData(sceneGPU.spot_light_count);
    for (size_t i = 0; i < sceneGPU.spot_light_count; ++i) {
      spotGpuData[i].position = glm::vec4(spotLights[i].position, 1.0f);
      glm::vec3 dir = spotLights[i].direction;
      if (glm::length(dir) < 0.0001f) {
        dir = glm::vec3(0.0f, -1.0f, 0.0f);
      }
      spotGpuData[i].direction = glm::vec4(glm::normalize(dir), 0.0f);
      spotGpuData[i].color =
          glm::vec4(spotLights[i].color, spotLights[i].intensity);

      float innerCos = glm::cos(glm::radians(spotLights[i].innerAngle));
      float outerCos = glm::cos(glm::radians(
          std::max(spotLights[i].outerAngle, spotLights[i].innerAngle + 0.1f)));
      spotGpuData[i].angles =
          glm::vec4(innerCos, outerCos, spotLights[i].innerAngle,
                    spotLights[i].outerAngle);
    }
    memcpy(spotLightsBufferMapped, spotGpuData.data(),
           sizeof(SpotLightGPU) * spotGpuData.size());
  }

  // =========================================================================
  // Заполнение ModelUniform (Set 1, Binding 0)
  // =========================================================================
  glm::mat4 model = glm::mat4(1.0f);
  model = glm::translate(model, cubePosition);
  model = glm::rotate(model, glm::radians(cubeRotation.x), glm::vec3(1, 0, 0));
  model = glm::rotate(model, glm::radians(cubeRotation.y), glm::vec3(0, 1, 0));
  model = glm::rotate(model, glm::radians(cubeRotation.z), glm::vec3(0, 0, 1));
  model = glm::scale(model, cubeScale);

  ModelUniformGPU modelGPU{};
  modelGPU.model = model;
  modelGPU.albedo = materialAlbedo;
  modelGPU.specular = materialSpecular;
  modelGPU.alpha = materialShininess;

  memcpy(modelUniformBufferMapped, &modelGPU, sizeof(modelGPU));

  // =========================================================================
  // Привязка буферов и наборов дескрипторов (Set 0 и Set 1)
  // =========================================================================
  VkBuffer vertexBuffers[] = {vertexBuffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertexBuffers, offsets);
  vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

  std::array<VkDescriptorSet, 2> descriptorSets = {sceneDescriptorSet,
                                                   modelDescriptorSet};
  vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelineLayout, 0,
                          static_cast<uint32_t>(descriptorSets.size()),
                          descriptorSets.data(), 0, nullptr);

  // Отрисовка куба с процедурными нормалями
  vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()), 1,
                   0, 0, 0);

  vkCmdEndRenderPass(fd.command_buffer);
  vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application