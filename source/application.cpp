#include "application.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace application {

namespace {

// =========================================================================
// Геометрия с текстурными координатами UV
// =========================================================================
struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec2 texCoord;
};

std::vector<Vertex> sceneVertices;
std::vector<uint16_t> sceneIndices;

uint32_t cubeIndexCount = 0;
uint32_t floorIndexOffset = 0;
uint32_t floorIndexCount = 0;

void generateCube(std::vector<Vertex> &verts, std::vector<uint16_t> &inds) {
  struct Face {
    glm::vec3 p0, p1, p2, p3;
  };

  const std::array<Face, 6> faces = {{
      // Передняя (+Z)
      {{-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}},
      // Задняя (-Z)
      {{0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}},
      // Верхняя (+Y)
      {{-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}},
      // Нижняя (-Y)
      {{-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}},
      // Правая (+X)
      {{0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}},
      // Левая (-X)
      {{-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, -0.5f}},
  }};

  for (const auto &face : faces) {
    auto start = static_cast<uint16_t>(verts.size());
    verts.push_back({face.p0, glm::vec3(0.0f), {0.0f, 0.0f}});
    verts.push_back({face.p1, glm::vec3(0.0f), {1.0f, 0.0f}});
    verts.push_back({face.p2, glm::vec3(0.0f), {1.0f, 1.0f}});
    verts.push_back({face.p3, glm::vec3(0.0f), {0.0f, 1.0f}});

    inds.push_back(start + 0); inds.push_back(start + 1); inds.push_back(start + 2);
    inds.push_back(start + 2); inds.push_back(start + 3); inds.push_back(start + 0);
  }
}

void generateFloor(std::vector<Vertex> &verts, std::vector<uint16_t> &inds) {
  auto start = static_cast<uint16_t>(verts.size());
  float y = -0.5f;
  float size = 5.0f;
  verts.push_back({{-size, y,  size}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}});
  verts.push_back({{ size, y,  size}, {0.0f, 1.0f, 0.0f}, {5.0f, 0.0f}});
  verts.push_back({{ size, y, -size}, {0.0f, 1.0f, 0.0f}, {5.0f, 5.0f}});
  verts.push_back({{-size, y, -size}, {0.0f, 1.0f, 0.0f}, {0.0f, 5.0f}});

  inds.push_back(start + 0); inds.push_back(start + 1); inds.push_back(start + 2);
  inds.push_back(start + 2); inds.push_back(start + 3); inds.push_back(start + 0);
}

void computeNormals(std::vector<Vertex> &verts, const std::vector<uint16_t> &inds) {
  for (size_t i = 0; i < inds.size(); i += 3) {
    uint16_t i0 = inds[i];
    uint16_t i1 = inds[i + 1];
    uint16_t i2 = inds[i + 2];
    glm::vec3 n = glm::cross(verts[i1].pos - verts[i0].pos, verts[i2].pos - verts[i0].pos);
    verts[i0].normal += n;
    verts[i1].normal += n;
    verts[i2].normal += n;
  }
  for (auto &v : verts) {
    if (glm::length(v.normal) > 0.0001f) v.normal = glm::normalize(v.normal);
  }
}

// =========================================================================
// GPU Структуры
// =========================================================================
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
static_assert(sizeof(SceneUniformsGPU) == 144, "SceneUniformsGPU layout mismatch");

struct ModelUniformGPU {
  glm::mat4 model;
  alignas(16) glm::vec3 albedo;
  alignas(16) glm::vec3 specular;
  float alpha;
};
static_assert(sizeof(ModelUniformGPU) == 96, "ModelUniformGPU layout mismatch");

struct PointLightGPU {
  glm::vec4 position;
  glm::vec4 color;
};

struct SpotLightGPU {
  glm::vec4 position;
  glm::vec4 direction;
  glm::vec4 color;
  glm::vec4 angles;
};

struct PushConstantsData {
  int32_t enable_special_sampling{0};
  float time{0.0f};
};

constexpr uint32_t MAX_LIGHTS = 32;
constexpr uint32_t MAX_MODELS = 16;

// =========================================================================
// Текстурные ресурсы
// =========================================================================
struct TextureResource {
  VkImage image{VK_NULL_HANDLE};
  VmaAllocation allocation{VK_NULL_HANDLE};
  VkImageView view{VK_NULL_HANDLE};
  VkSampler sampler{VK_NULL_HANDLE};
};

TextureResource blackTexture;
std::vector<TextureResource> loadedTextures;
std::vector<std::string> textureNames;

TextureResource crateSpecular;
TextureResource crateEmissive;

// Staging & Device Memory
VkBuffer stagingBuffer = VK_NULL_HANDLE;
VmaAllocation stagingBufferAllocation = VK_NULL_HANDLE;
uint8_t *stagingBufferMemory = nullptr;
VkDeviceSize stagingBufferOffset = 0;

VkBuffer vertexBufferGPU = VK_NULL_HANDLE;
VmaAllocation vertexBufferAllocation = VK_NULL_HANDLE;
VkBuffer indexBufferGPU = VK_NULL_HANDLE;
VmaAllocation indexBufferAllocation = VK_NULL_HANDLE;

// Dynamic UBO
VkBuffer dynamicModelUniformBuffer = VK_NULL_HANDLE;
VmaAllocation dynamicModelUniformBufferAllocation = VK_NULL_HANDLE;
uint8_t *dynamicModelUniformBufferMapped = nullptr;
VkDeviceSize dynamicBufferAlignment = 0;
VkDeviceSize dynamicModelUniformSizeAligned = 0;

// Scene & Light Buffers
VkBuffer sceneUniformBuffer = VK_NULL_HANDLE;
VmaAllocation sceneUniformBufferAllocation = VK_NULL_HANDLE;
void *sceneUniformBufferMapped = nullptr;

VkBuffer pointLightsBuffer = VK_NULL_HANDLE;
VmaAllocation pointLightsBufferAllocation = VK_NULL_HANDLE;
void *pointLightsBufferMapped = nullptr;

VkBuffer spotLightsBuffer = VK_NULL_HANDLE;
VmaAllocation spotLightsBufferAllocation = VK_NULL_HANDLE;
void *spotLightsBufferMapped = nullptr;

// Layouts, Pools, Sets
VkDescriptorSetLayout sceneDescriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelDescriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;
std::vector<VkDescriptorSet> modelDescriptorSets;

VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline graphicsPipeline = VK_NULL_HANDLE;

VkCommandPool uploadCommandPool = VK_NULL_HANDLE;

// Камера и сцена
glm::vec3 cameraPos = glm::vec3(0.0f, 1.8f, 4.0f);
float cameraYaw = -90.0f;
float cameraPitch = -20.0f;
float fov = 45.0f;

glm::vec3 ambientLightColor = glm::vec3(0.08f);
bool dirLightEnabled = true;
glm::vec3 dirLightDirection = glm::vec3(-0.6f, -1.0f, -0.5f);
glm::vec3 dirLightColor = glm::vec3(1.0f, 0.95f, 0.85f);
float dirLightIntensity = 1.5f;

struct ModelInstance {
  glm::vec3 position{0.0f};
  glm::vec3 rotation{0.0f};
  glm::vec3 scale{1.0f};
  glm::vec3 albedoTint{1.0f};
  glm::vec3 specularTint{1.0f};
  float shininess{64.0f};
  int materialIndex{0};
  bool specialSampling{false};
};

std::vector<ModelInstance> sceneModels;

// =========================================================================
// Вспомогательные функции
// =========================================================================
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
  VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule shaderModule;
  if (vkCreateShaderModule(graphics::internal::context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }
  return shaderModule;
}

void createUploadCommandPool() {
  auto &ctx = graphics::internal::context;
  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.queueFamilyIndex = ctx.graphics_queue_index;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

  if (vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &uploadCommandPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create upload command pool!");
  }
}

VkCommandBuffer beginSingleTimeCommands() {
  auto &ctx = graphics::internal::context;
  VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = uploadCommandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  if (vkAllocateCommandBuffers(ctx.device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate single time command buffer!");
  }

  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(commandBuffer, &beginInfo);
  return commandBuffer;
}

void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
  auto &ctx = graphics::internal::context;
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  VkFence fence;
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(ctx.device, &fenceInfo, nullptr, &fence);

  vkQueueSubmit(ctx.graphics_queue, 1, &submitInfo, fence);
  vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);

  vkDestroyFence(ctx.device, fence, nullptr);
  vkFreeCommandBuffers(ctx.device, uploadCommandPool, 1, &commandBuffer);
}

// =========================================================================
// Загрузка текстур
// =========================================================================
TextureResource createTextureFromRGBA(uint32_t width, uint32_t height, const uint8_t *pixels) {
  auto &ctx = graphics::internal::context;
  TextureResource res{};

  VkDeviceSize imageSize = width * height * 4;

  VkDeviceSize offset = stagingBufferOffset;
  memcpy(stagingBufferMemory + offset, pixels, static_cast<size_t>(imageSize));
  stagingBufferOffset += imageSize;

  VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imgInfo.imageType = VK_IMAGE_TYPE_2D;
  imgInfo.extent.width = width;
  imgInfo.extent.height = height;
  imgInfo.extent.depth = 1;
  imgInfo.mipLevels = 1;
  imgInfo.arrayLayers = 1;
  imgInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;

  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  if (vmaCreateImage(ctx.allocator, &imgInfo, &allocInfo, &res.image, &res.allocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image!");
  }

  VkCommandBuffer cmd = beginSingleTimeCommands();

  VkImageMemoryBarrier b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  b1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b1.image = res.image;
  b1.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  b1.subresourceRange.baseMipLevel = 0;
  b1.subresourceRange.levelCount = 1;
  b1.subresourceRange.baseArrayLayer = 0;
  b1.subresourceRange.layerCount = 1;
  b1.srcAccessMask = 0;
  b1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                       0, nullptr, 0, nullptr, 1, &b1);

  VkBufferImageCopy region{};
  region.bufferOffset = offset;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {width, height, 1};

  vkCmdCopyBufferToImage(cmd, stagingBuffer, res.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  VkImageMemoryBarrier b2 = b1;
  b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                       0, nullptr, 0, nullptr, 1, &b2);

  endSingleTimeCommands(cmd);

  // ImageView
  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.image = res.image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.layerCount = 1;
  if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &res.view) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image view!");
  }

  // Sampler
  VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  if (vkCreateSampler(ctx.device, &samplerInfo, nullptr, &res.sampler) != VK_SUCCESS) {
    throw std::runtime_error("failed to create sampler!");
  }

  return res;
}

// Загрузка текстуры из файла на диске (PNG/JPG/BMP/TGA)
bool loadTextureFromFile(const std::string &filepath, TextureResource &outTex) {
  int width = 0, height = 0, channels = 0;
  stbi_uc *pixels = stbi_load(filepath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
  if (!pixels) {
    return false;
  }

  outTex = createTextureFromRGBA(static_cast<uint32_t>(width), static_cast<uint32_t>(height), pixels);
  stbi_image_free(pixels);
  return true;
}

// Процедурные текстуры для фоллбэка и эффектов
std::vector<uint8_t> generateCheckerPattern(uint32_t w, uint32_t h) {
  std::vector<uint8_t> p(w * h * 4);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      bool c = ((x / 32) ^ (y / 32)) & 1;
      uint8_t val = c ? 220 : 60;
      size_t idx = (y * w + x) * 4;
      p[idx + 0] = val; p[idx + 1] = val; p[idx + 2] = val; p[idx + 3] = 255;
    }
  }
  return p;
}

std::vector<uint8_t> generateCratePattern(uint32_t w, uint32_t h) {
  std::vector<uint8_t> p(w * h * 4);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      size_t idx = (y * w + x) * 4;
      bool border = (x < 16 || x >= w - 16 || y < 16 || y >= h - 16 || std::abs(int(x) - int(y)) < 12);
      if (border) {
        p[idx + 0] = 130; p[idx + 1] = 90; p[idx + 2] = 50; p[idx + 3] = 255;
      } else {
        p[idx + 0] = 190; p[idx + 1] = 145; p[idx + 2] = 95; p[idx + 3] = 255;
      }
    }
  }
  return p;
}

std::vector<uint8_t> generateEmissiveRunes(uint32_t w, uint32_t h) {
  std::vector<uint8_t> p(w * h * 4, 0);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      size_t idx = (y * w + x) * 4;
      float fx = (float)x / w - 0.5f;
      float fy = (float)y / h - 0.5f;
      float r = std::sqrt(fx * fx + fy * fy);
      if (std::abs(r - 0.25f) < 0.03f || (std::abs(fx) < 0.02f && std::abs(fy) < 0.2f)) {
        p[idx + 0] = 30;
        p[idx + 1] = 230;
        p[idx + 2] = 255;
        p[idx + 3] = 255;
      }
    }
  }
  return p;
}

// Контрастная шахматная маска Specular: белые квадраты дают 100% блик, черные — матовые
std::vector<uint8_t> generateSpecularChecker(uint32_t w, uint32_t h) {
  std::vector<uint8_t> p(w * h * 4);
  const uint32_t tileSize = 32;
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      bool shiny = ((x / tileSize) ^ (y / tileSize)) & 1;
      uint8_t val = shiny ? 255 : 0;
      size_t idx = (y * w + x) * 4;
      p[idx + 0] = val; // R
      p[idx + 1] = val; // G
      p[idx + 2] = val; // B
      p[idx + 3] = 255; // A
    }
  }
  return p;
}

void destroyTextureResource(TextureResource &t) {
  auto &ctx = graphics::internal::context;
  if (t.sampler) { vkDestroySampler(ctx.device, t.sampler, nullptr); t.sampler = VK_NULL_HANDLE; }
  if (t.view) { vkDestroyImageView(ctx.device, t.view, nullptr); t.view = VK_NULL_HANDLE; }
  if (t.image) { vmaDestroyImage(ctx.allocator, t.image, t.allocation); t.image = VK_NULL_HANDLE; }
}

} // namespace

bool initialize() {
  auto &ctx = graphics::internal::context;

  // 1. Создаем командный пул для staging-операций[cite: 2]
  createUploadCommandPool();

  // 2. Расчет выравнивания динамического UBO[cite: 2]
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
  dynamicBufferAlignment = props.limits.minUniformBufferOffsetAlignment;
  dynamicModelUniformSizeAligned =
      (sizeof(ModelUniformGPU) + dynamicBufferAlignment - 1) & ~(dynamicBufferAlignment - 1);

  // 3. Staging-буфер на 256 МБ для больших текстур[cite: 2]
  VkBufferCreateInfo stagingInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  stagingInfo.size = 256ULL * 1024 * 1024;
  stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo stagingAlloc{
      .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO,
  };
  vmaCreateBuffer(ctx.allocator, &stagingInfo, &stagingAlloc, &stagingBuffer, &stagingBufferAllocation, nullptr);
  vmaMapMemory(ctx.allocator, stagingBufferAllocation, reinterpret_cast<void **>(&stagingBufferMemory));

  // 4. Геометрия сцены[cite: 2]
  sceneVertices.clear();
  sceneIndices.clear();

  generateCube(sceneVertices, sceneIndices);
  cubeIndexCount = static_cast<uint32_t>(sceneIndices.size());

  floorIndexOffset = cubeIndexCount;
  generateFloor(sceneVertices, sceneIndices);
  floorIndexCount = static_cast<uint32_t>(sceneIndices.size()) - cubeIndexCount;

  computeNormals(sceneVertices, sceneIndices);

  // 5. Перенос вершинного и индексного буфера в Device Local VRAM[cite: 2]
  VkDeviceSize vSize = sizeof(Vertex) * sceneVertices.size();
  VkDeviceSize iSize = sizeof(uint16_t) * sceneIndices.size();

  VkDeviceSize vOffset = stagingBufferOffset;
  memcpy(stagingBufferMemory + vOffset, sceneVertices.data(), vSize);
  stagingBufferOffset += vSize;

  VkDeviceSize iOffset = stagingBufferOffset;
  memcpy(stagingBufferMemory + iOffset, sceneIndices.data(), iSize);
  stagingBufferOffset += iSize;

  VkBufferCreateInfo vBufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  vBufInfo.size = vSize;
  vBufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  vBufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo gpuAlloc{.usage = VMA_MEMORY_USAGE_GPU_ONLY};
  vmaCreateBuffer(ctx.allocator, &vBufInfo, &gpuAlloc, &vertexBufferGPU, &vertexBufferAllocation, nullptr);

  VkBufferCreateInfo iBufInfo = vBufInfo;
  iBufInfo.size = iSize;
  iBufInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  vmaCreateBuffer(ctx.allocator, &iBufInfo, &gpuAlloc, &indexBufferGPU, &indexBufferAllocation, nullptr);

  VkCommandBuffer cmd = beginSingleTimeCommands();
  VkBufferCopy vCopy{.srcOffset = vOffset, .dstOffset = 0, .size = vSize};
  vkCmdCopyBuffer(cmd, stagingBuffer, vertexBufferGPU, 1, &vCopy);

  VkBufferCopy iCopy{.srcOffset = iOffset, .dstOffset = 0, .size = iSize};
  vkCmdCopyBuffer(cmd, stagingBuffer, indexBufferGPU, 1, &iCopy);
  endSingleTimeCommands(cmd);

  // 6. Текстуры (Albedo, Specular, Emissive)[cite: 2]
  uint8_t blackPixel[4] = {0, 0, 0, 255};
  blackTexture = createTextureFromRGBA(1, 1, blackPixel);

  loadedTextures.clear();
  textureNames.clear();

  const std::vector<std::pair<std::string, std::string>> filesToLoad = {
      {"textures/cat.png", "Cat"},
      {"textures/crate.png", "Crate"},
      {"textures/checker.png", "Checker Floor"},
      {"textures/brick.jpg", "Brick Wall"},
      {"textures/wood.jpg", "Wood Planks"},
  };

  for (const auto &[path, label] : filesToLoad) {
    TextureResource tex;
    if (loadTextureFromFile(path, tex)) {
      loadedTextures.push_back(tex);
      textureNames.push_back(label + " (File)");
    }
  }

  // Фоллбэк при отсутствии файлов
  if (loadedTextures.empty()) {
    auto cratePixels = generateCratePattern(256, 256);
    auto checkerPixels = generateCheckerPattern(256, 256);

    loadedTextures.push_back(createTextureFromRGBA(256, 256, cratePixels.data()));
    textureNames.push_back("Crate (Procedural)");

    loadedTextures.push_back(createTextureFromRGBA(256, 256, checkerPixels.data()));
    textureNames.push_back("Checker Floor (Procedural)");
  }

  // Specular
  TextureResource fileSpecular;
  if (loadTextureFromFile("textures/specular.png", fileSpecular) ||
      loadTextureFromFile("textures/specular.jpg", fileSpecular)) {
    crateSpecular = fileSpecular;
  } else {
    auto specPixels = generateSpecularChecker(256, 256);
    crateSpecular = createTextureFromRGBA(256, 256, specPixels.data());
  }

  // Emissive
  TextureResource fileEmissive;
  if (loadTextureFromFile("textures/emissive.jpg", fileEmissive) ||
      loadTextureFromFile("textures/emissive.png", fileEmissive)) {
    crateEmissive = fileEmissive;
  } else {
    auto emissivePixels = generateEmissiveRunes(256, 256);
    crateEmissive = createTextureFromRGBA(256, 256, emissivePixels.data());
  }

  // 7. Dynamic Model Uniform Buffer[cite: 2]
  VkBufferCreateInfo dynUboInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  dynUboInfo.size = dynamicModelUniformSizeAligned * MAX_MODELS;
  dynUboInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  dynUboInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo dynAlloc{
      .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO,
  };
  vmaCreateBuffer(ctx.allocator, &dynUboInfo, &dynAlloc, &dynamicModelUniformBuffer,
                  &dynamicModelUniformBufferAllocation, nullptr);
  vmaMapMemory(ctx.allocator, dynamicModelUniformBufferAllocation,
               reinterpret_cast<void **>(&dynamicModelUniformBufferMapped));

  // 8. Буферы Сцены и Источников света[cite: 2]
  VkBufferCreateInfo sceneUboInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  sceneUboInfo.size = sizeof(SceneUniformsGPU);
  sceneUboInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  vmaCreateBuffer(ctx.allocator, &sceneUboInfo, &dynAlloc, &sceneUniformBuffer, &sceneUniformBufferAllocation, nullptr);
  vmaMapMemory(ctx.allocator, sceneUniformBufferAllocation, &sceneUniformBufferMapped);

  VkBufferCreateInfo ssboInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ssboInfo.size = sizeof(PointLightGPU) * MAX_LIGHTS;
  ssboInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  vmaCreateBuffer(ctx.allocator, &ssboInfo, &dynAlloc, &pointLightsBuffer, &pointLightsBufferAllocation, nullptr);
  vmaMapMemory(ctx.allocator, pointLightsBufferAllocation, &pointLightsBufferMapped);

  ssboInfo.size = sizeof(SpotLightGPU) * MAX_LIGHTS;
  vmaCreateBuffer(ctx.allocator, &ssboInfo, &dynAlloc, &spotLightsBuffer, &spotLightsBufferAllocation, nullptr);
  vmaMapMemory(ctx.allocator, spotLightsBufferAllocation, &spotLightsBufferMapped);

  // 9. Descriptor Set Layouts (Set 0 и Set 1)[cite: 2]
  std::array<VkDescriptorSetLayoutBinding, 3> sceneBindings{};
  sceneBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  sceneBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  sceneBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

  VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  sceneLayoutInfo.bindingCount = static_cast<uint32_t>(sceneBindings.size());
  sceneLayoutInfo.pBindings = sceneBindings.data();
  vkCreateDescriptorSetLayout(ctx.device, &sceneLayoutInfo, nullptr, &sceneDescriptorSetLayout);

  std::array<VkDescriptorSetLayoutBinding, 4> modelBindings{};
  modelBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  modelBindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  modelBindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  modelBindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

  VkDescriptorSetLayoutCreateInfo modelLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  modelLayoutInfo.bindingCount = static_cast<uint32_t>(modelBindings.size());
  modelLayoutInfo.pBindings = modelBindings.data();
  vkCreateDescriptorSetLayout(ctx.device, &modelLayoutInfo, nullptr, &modelDescriptorSetLayout);

  // 10. Descriptor Pool & Sets[cite: 2]
  std::array<VkDescriptorPoolSize, 4> poolSizes{{
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 16},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
  }};

  VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = 32;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool);

  // Set 0 Allocation
  VkDescriptorSetAllocateInfo allocScene{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocScene.descriptorPool = descriptorPool;
  allocScene.descriptorSetCount = 1;
  allocScene.pSetLayouts = &sceneDescriptorSetLayout;
  vkAllocateDescriptorSets(ctx.device, &allocScene, &sceneDescriptorSet);

  VkDescriptorBufferInfo scInfo{sceneUniformBuffer, 0, sizeof(SceneUniformsGPU)};
  VkDescriptorBufferInfo ptInfo{pointLightsBuffer, 0, sizeof(PointLightGPU) * MAX_LIGHTS};
  VkDescriptorBufferInfo spInfo{spotLightsBuffer, 0, sizeof(SpotLightGPU) * MAX_LIGHTS};

  std::array<VkWriteDescriptorSet, 3> scWrites{};
  scWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDescriptorSet, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &scInfo, nullptr};
  scWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDescriptorSet, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ptInfo, nullptr};
  scWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDescriptorSet, 2, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spInfo, nullptr};
  vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(scWrites.size()), scWrites.data(), 0, nullptr);

  // Set 1 Allocation под каждую текстуру[cite: 2]
  modelDescriptorSets.resize(loadedTextures.size());
  for (size_t m = 0; m < modelDescriptorSets.size(); ++m) {
    VkDescriptorSetAllocateInfo allocModel{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocModel.descriptorPool = descriptorPool;
    allocModel.descriptorSetCount = 1;
    allocModel.pSetLayouts = &modelDescriptorSetLayout;
    vkAllocateDescriptorSets(ctx.device, &allocModel, &modelDescriptorSets[m]);

    VkDescriptorBufferInfo dynInfo{dynamicModelUniformBuffer, 0, sizeof(ModelUniformGPU)};
    VkDescriptorImageInfo albInfo{loadedTextures[m].sampler, loadedTextures[m].view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo spcInfo{crateSpecular.sampler, crateSpecular.view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                                  
    // crateEmissive подключена ко ВСЕМ материалам (включая jpg и png):
    VkDescriptorImageInfo emiInfo{crateEmissive.sampler, crateEmissive.view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    std::array<VkWriteDescriptorSet, 4> mWrites{};
    mWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelDescriptorSets[m], 0, 0, 1,
                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &dynInfo, nullptr};
    mWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelDescriptorSets[m], 1, 0, 1,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &albInfo, nullptr, nullptr};
    mWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelDescriptorSets[m], 2, 0, 1,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &spcInfo, nullptr, nullptr};
    mWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelDescriptorSets[m], 3, 0, 1,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &emiInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(mWrites.size()), mWrites.data(), 0, nullptr);
  }

  // 11. Pipeline Layout с Push Constants[cite: 2]
  VkPushConstantRange pushConstRange{};
  pushConstRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstRange.offset = 0;
  pushConstRange.size = sizeof(PushConstantsData);

  std::array<VkDescriptorSetLayout, 2> setLayouts = {sceneDescriptorSetLayout, modelDescriptorSetLayout};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
  pipelineLayoutInfo.pSetLayouts = setLayouts.data();
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstRange;
  if (vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  // 12. Pipeline[cite: 2]
  auto vertCode = readFile("shaders/cube.vert.spv");
  auto fragCode = readFile("shaders/cube.frag.spv");
  VkShaderModule vertMod = createShaderModule(vertCode);
  VkShaderModule fragMod = createShaderModule(fragCode);

  VkPipelineShaderStageCreateInfo stages[2]{
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertMod, "main", nullptr},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "main", nullptr},
  };

  VkVertexInputBindingDescription bindDesc{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
  std::array<VkVertexInputAttributeDescription, 3> attrDesc{{
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
      {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord)},
  }};

  VkPipelineVertexInputStateCreateInfo vInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vInput.vertexBindingDescriptionCount = 1;
  vInput.pVertexBindingDescriptions = &bindDesc;
  vInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
  vInput.pVertexAttributeDescriptions = attrDesc.data();

  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1;
  vp.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rast.polygonMode = VK_POLYGON_MODE_FILL;
  rast.cullMode = VK_CULL_MODE_BACK_BIT;
  rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rast.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = VK_TRUE;
  ds.depthWriteEnable = VK_TRUE;
  ds.depthCompareOp = VK_COMPARE_OP_LESS;

  VkPipelineColorBlendAttachmentState cbAtt{};
  cbAtt.colorWriteMask = 0xF;

  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1;
  cb.pAttachments = &cbAtt;

  std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynStateInfo{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynStateInfo.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
  dynStateInfo.pDynamicStates = dynStates.data();

  VkGraphicsPipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipeInfo.stageCount = 2;
  pipeInfo.pStages = stages;
  pipeInfo.pVertexInputState = &vInput;
  pipeInfo.pInputAssemblyState = &ia;
  pipeInfo.pViewportState = &vp;
  pipeInfo.pRasterizationState = &rast;
  pipeInfo.pMultisampleState = &ms;
  pipeInfo.pDepthStencilState = &ds;
  pipeInfo.pColorBlendState = &cb;
  pipeInfo.pDynamicState = &dynStateInfo;
  pipeInfo.layout = pipelineLayout;
  pipeInfo.renderPass = ctx.render_pass;

  if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }

  vkDestroyShaderModule(ctx.device, fragMod, nullptr);
  vkDestroyShaderModule(ctx.device, vertMod, nullptr);

  // Инициализация моделей сцены
  sceneModels.clear();
  ModelInstance cube{};
  cube.position = glm::vec3(0.0f, 0.0f, 0.0f);
  cube.materialIndex = 0;
  cube.specialSampling = false;
  cube.albedoTint = glm::vec3(1.0f);
  cube.specularTint = glm::vec3(1.0f, 1.0f, 1.0f);
  cube.shininess = 64.0f;
  sceneModels.push_back(cube);

  ModelInstance floor{};
  floor.position = glm::vec3(0.0f, 0.0f, 0.0f);
  floor.materialIndex = static_cast<int>(std::min(size_t(1), loadedTextures.size() - 1));
  floor.albedoTint = glm::vec3(1.0f);
  floor.specularTint = glm::vec3(0.0f);
  floor.shininess = 16.0f;
  sceneModels.push_back(floor);

  return true;
}

void shutdown() {
  auto &ctx = graphics::internal::context;
  vkQueueWaitIdle(ctx.graphics_queue);

  vkDestroyPipeline(ctx.device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);

  destroyTextureResource(blackTexture);
  destroyTextureResource(crateSpecular);
  destroyTextureResource(crateEmissive);
  for (auto &t : loadedTextures) destroyTextureResource(t);

  vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(ctx.device, modelDescriptorSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(ctx.device, sceneDescriptorSetLayout, nullptr);

  vmaUnmapMemory(ctx.allocator, dynamicModelUniformBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, dynamicModelUniformBuffer, dynamicModelUniformBufferAllocation);

  vmaUnmapMemory(ctx.allocator, spotLightsBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, spotLightsBuffer, spotLightsBufferAllocation);

  vmaUnmapMemory(ctx.allocator, pointLightsBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, pointLightsBuffer, pointLightsBufferAllocation);

  vmaUnmapMemory(ctx.allocator, sceneUniformBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, sceneUniformBuffer, sceneUniformBufferAllocation);

  vmaDestroyBuffer(ctx.allocator, indexBufferGPU, indexBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, vertexBufferGPU, vertexBufferAllocation);

  vmaUnmapMemory(ctx.allocator, stagingBufferAllocation);
  vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingBufferAllocation);

  if (uploadCommandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(ctx.device, uploadCommandPool, nullptr);
    uploadCommandPool = VK_NULL_HANDLE;
  }
}

void update(double time) {
  ImGui::SetNextWindowSize(ImVec2(480, 680), ImGuiCond_FirstUseEver);
  ImGui::Begin("Lab #3: Texturing & Specular Test");

  if (ImGui::CollapsingHeader("Camera Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::DragFloat3("Position", glm::value_ptr(cameraPos), 0.05f);
    ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Pitch", &cameraPitch, -89.0f, 89.0f, "%.1f deg");
    ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f, "%.1f deg");
  }

  if (ImGui::CollapsingHeader("Lighting (Blinn-Phong Source)", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::ColorEdit3("Ambient Light", glm::value_ptr(ambientLightColor));
    ImGui::Checkbox("Directional Light", &dirLightEnabled);
    if (dirLightEnabled) {
      ImGui::SliderFloat3("Dir Direction", glm::value_ptr(dirLightDirection), -1.0f, 1.0f);
      ImGui::ColorEdit3("Dir Color", glm::value_ptr(dirLightColor));
      ImGui::SliderFloat("Dir Intensity", &dirLightIntensity, 0.0f, 5.0f, "%.2f");
    }
  }

  auto drawTextureSelector = [&](const char *label, int *matIdx) {
    if (ImGui::BeginCombo(label, textureNames[*matIdx].c_str())) {
      for (size_t i = 0; i < textureNames.size(); ++i) {
        bool isSelected = (*matIdx == static_cast<int>(i));
        if (ImGui::Selectable(textureNames[i].c_str(), isSelected)) {
          *matIdx = static_cast<int>(i);
        }
        if (isSelected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
  };

  if (ImGui::CollapsingHeader("Model 1: Interactive Cube", ImGuiTreeNodeFlags_DefaultOpen)) {
    drawTextureSelector("Texture / Material", &sceneModels[0].materialIndex);
    ImGui::Checkbox("Enable Non-Trivial Sampling (Wave)", &sceneModels[0].specialSampling);
    ImGui::DragFloat3("Position", glm::value_ptr(sceneModels[0].position), 0.05f);
    ImGui::DragFloat3("Rotation", glm::value_ptr(sceneModels[0].rotation), 1.0f);

    ImGui::Separator();
    ImGui::Text("Material Properties (Blinn-Phong):");
    ImGui::ColorEdit3("Albedo Tint", glm::value_ptr(sceneModels[0].albedoTint));
    ImGui::ColorEdit3("Specular Tint", glm::value_ptr(sceneModels[0].specularTint));
    ImGui::SliderFloat("Shininess", &sceneModels[0].shininess, 1.0f, 256.0f, "%.1f");
  }

  if (ImGui::CollapsingHeader("Model 2: Floor", ImGuiTreeNodeFlags_DefaultOpen)) {
    drawTextureSelector("Floor Material", &sceneModels[1].materialIndex);
    ImGui::ColorEdit3("Floor Tint", glm::value_ptr(sceneModels[1].albedoTint));
  }

  static bool autoRotate = true;
  ImGui::Checkbox("Auto-Rotate Cube", &autoRotate);
  if (autoRotate) {
    sceneModels[0].rotation.y = static_cast<float>(time) * 20.0f;
  }

  ImGui::End();
}

void render(const graphics::internal::FrameData &fd) {
  auto &ctx = graphics::internal::context;

  vkResetCommandBuffer(fd.command_buffer, 0);
  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                     VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
  vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

  VkClearValue clearValues[2]{};
  clearValues[0].color = {{0.03f, 0.03f, 0.05f, 1.0f}};
  clearValues[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo rpInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rpInfo.renderPass = ctx.render_pass;
  rpInfo.framebuffer = fd.framebuffer;
  rpInfo.renderArea.extent = ctx.swapchain_extent;
  rpInfo.clearValueCount = 2;
  rpInfo.pClearValues = clearValues;

  vkCmdBeginRenderPass(fd.command_buffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

  VkViewport vp{0.0f, 0.0f, (float)ctx.swapchain_extent.width, (float)ctx.swapchain_extent.height, 0.0f, 1.0f};
  vkCmdSetViewport(fd.command_buffer, 0, 1, &vp);

  VkRect2D scissor{{0, 0}, ctx.swapchain_extent};
  vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

  // 1. Камера и Scene Uniforms[cite: 2]
  glm::vec3 front;
  front.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
  front.y = sin(glm::radians(cameraPitch));
  front.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
  glm::vec3 cFront = glm::normalize(front);
  glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cFront, glm::vec3(0.0f, 1.0f, 0.0f));

  float aspect = vp.width / vp.height;
  glm::mat4 proj = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.0f);
  proj[1][1] *= -1;

  SceneUniformsGPU sceneGPU{};
  sceneGPU.view_projection = proj * view;
  sceneGPU.camera_position = cameraPos;
  sceneGPU.directional_light_direction = glm::normalize(dirLightDirection);
  sceneGPU.directional_light_color = dirLightEnabled ? (dirLightColor * dirLightIntensity) : glm::vec3(0.0f);
  sceneGPU.ambient_light = ambientLightColor;
  sceneGPU.point_light_count = 0;
  sceneGPU.spot_light_count = 0;
  memcpy(sceneUniformBufferMapped, &sceneGPU, sizeof(sceneGPU));

  vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                          &sceneDescriptorSet, 0, nullptr);

  // 2. Обновление массива моделей в Dynamic UBO[cite: 2]
  for (size_t i = 0; i < sceneModels.size(); ++i) {
    auto *dst = reinterpret_cast<ModelUniformGPU *>(dynamicModelUniformBufferMapped +
                                                    (i * dynamicModelUniformSizeAligned));
    glm::mat4 m{1.0f};
    m = glm::translate(m, sceneModels[i].position);
    m = glm::rotate(m, glm::radians(sceneModels[i].rotation.x), glm::vec3(1, 0, 0));
    m = glm::rotate(m, glm::radians(sceneModels[i].rotation.y), glm::vec3(0, 1, 0));
    m = glm::rotate(m, glm::radians(sceneModels[i].rotation.z), glm::vec3(0, 0, 1));
    m = glm::scale(m, sceneModels[i].scale);

    dst->model = m;
    dst->albedo = sceneModels[i].albedoTint;
    dst->specular = sceneModels[i].specularTint;
    dst->alpha = sceneModels[i].shininess;
  }

  // 3. Привязка буферов вершин и индексов[cite: 2]
  VkBuffer vBuffers[] = {vertexBufferGPU};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vBuffers, offsets);
  vkCmdBindIndexBuffer(fd.command_buffer, indexBufferGPU, 0, VK_INDEX_TYPE_UINT16);

  static float currentTime = 0.0f;
  currentTime += 0.016f;

  // 4. Отрисовка с динамическим смещением и push-константами[cite: 2]
  for (uint32_t i = 0; i < sceneModels.size(); ++i) {
    uint32_t dynamicOffset = static_cast<uint32_t>(i * dynamicModelUniformSizeAligned);
    int matIdx = sceneModels[i].materialIndex;

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1,
                            &modelDescriptorSets[matIdx], 1, &dynamicOffset);

    PushConstantsData pushData{};
    pushData.enable_special_sampling = sceneModels[i].specialSampling ? 1 : 0;
    pushData.time = currentTime;
    vkCmdPushConstants(fd.command_buffer, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstantsData), &pushData);

    if (i == 0) {
      vkCmdDrawIndexed(fd.command_buffer, cubeIndexCount, 1, 0, 0, 0);
    } else {
      vkCmdDrawIndexed(fd.command_buffer, floorIndexCount, 1, floorIndexOffset, 0, 0);
    }
  }

  vkCmdEndRenderPass(fd.command_buffer);
  vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application