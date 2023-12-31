
#ifdef __cplusplus
#pragma once
struct PushConstants {
    glm::mat4 invView;
    glm::mat4 invProj;
    int frame = 0;
    int sampleCount = 10;
    float bloomThreshold = 0.5f;
    float domeLightPhi = 0.0f;
    int enableNEE = 1;
    int _dummy[3];
    glm::vec4 infiniteLightDirection = glm::vec4{glm::normalize(glm::vec3{-1.0, -1.0, 0.3}), 1.0};
    float infiniteLightIntensity = 0.8;
};
#else
layout(push_constant) uniform PushConstants {
    mat4 invView;
    mat4 invProj;
    int frame;
    int sampleCount;
    float bloomThreshold;
    float domeLightPhi;
    int enableNEE;
    int _dummy[3];
    vec4 infiniteLightDirection;
    float infiniteLightIntensity;
};
#endif

#ifndef __cplusplus

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

const float PI = 3.1415926535;

struct HitPayload {
    vec3 radiance;
    int depth;
    uint seed;
};

struct Address {
    uint64_t materials;
};

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 texCoord;
};

struct Material {
    int baseColorTextureIndex;
    int metallicRoughnessTextureIndex;
    int normalTextureIndex;
    int occlusionTextureIndex;
    int emissiveTextureIndex;

    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    vec3 emissiveFactor;
};

// Image
layout(binding = 0, rgba32f) uniform image2D baseImage;
layout(binding = 1, rgba32f) uniform image2D bloomImage;

// Accel
layout(binding = 10) uniform accelerationStructureEXT topLevelAS;

// Buffer
layout(binding = 20) buffer VertexBuffers {
    float vertices[];
}
vertexBuffers[];

layout(binding = 21) buffer IndexBuffers {
    uint indices[];
}
indexBuffers[];

layout(binding = 22) buffer TransformMatrixBuffer {
    mat4 transformMatrices[];
};

layout(binding = 23) buffer NormalMatrixBuffer {
    mat4 normalMatrices[];
};

layout(binding = 24) buffer AddressBuffer {
    Address addresses;
};

layout(binding = 25) buffer MaterialIndexBuffer {
    int materialIndices[];
};

// Buffer reference
layout(buffer_reference, scalar) buffer Materials {
    Material materials[];
};

#endif
