#include <future>
#include <random>

#include "../shader/share.h"
#include "reactive/App.hpp"

#define NOMINMAX
#define TINYGLTF_IMPLEMENTATION
#include "reactive/common.hpp"
#include "render_pass.hpp"
#include "scene.hpp"

class Renderer {
public:
    Renderer(const Context& context, uint32_t width, uint32_t height, App* app)
        : width{width}, height{height} {
        // Output ray tracing props
        auto rtProps =
            context
                .getPhysicalDeviceProperties2<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        maxRayRecursionDepth = rtProps.maxRayRecursionDepth;
        spdlog::info("MaxRayRecursionDepth: {}", maxRayRecursionDepth);

        rv::CPUTimer timer;
        scene.loadFromFile(context);
        spdlog::info("Load from file: {} ms", timer.elapsedInMilli());

        timer.restart();
        scene.buildAccels(context);
        spdlog::info("Build accels: {} ms", timer.elapsedInMilli());

        // Add materials
        Material diffuseMaterial;
        diffuseMaterial.baseColorFactor = glm::vec4{1.0, 1.0, 1.0, 1.0};
        Material glassMaterial;
        glassMaterial.baseColorFactor = glm::vec4{1.0, 1.0, 1.0, 0.0};
        glassMaterial.roughnessFactor = 0.1;
        Material metalMaterial;
        metalMaterial.baseColorFactor = glm::vec4{1.0, 1.0, 1.0, 1.0};
        metalMaterial.metallicFactor = 1.0;
        metalMaterial.roughnessFactor = 0.2;
        int diffuseMaterialIndex = scene.addMaterial(context, diffuseMaterial);
        int glassMaterialIndex = scene.addMaterial(context, glassMaterial);
        int metalMaterialIndex = scene.addMaterial(context, metalMaterial);

        // Set materials
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist1(0.0f, 1.0f);
        for (auto& materialIndex : scene.materialIndices) {
            if (materialIndex == -1) {
                double randVal = dist1(rng);
                if (randVal < 0.33) {
                    materialIndex = diffuseMaterialIndex;
                } else if (randVal < 0.66) {
                    materialIndex = metalMaterialIndex;
                } else {
                    materialIndex = glassMaterialIndex;
                }
            }
        }

        scene.initMaterialIndexBuffer(context);
        scene.initAddressBuffer(context);

        baseImage = context.createImage({
            .usage = ImageUsage::Storage,
            .extent = {width, height, 1},
            .format = vk::Format::eR32G32B32A32Sfloat,
            .layout = vk::ImageLayout::eGeneral,
            .debugName = "baseImage",
        });

        createPipelines(context);

        orbitalCamera = OrbitalCamera{app, width, height};
        orbitalCamera.phi = 25.0f;
        orbitalCamera.theta = 30.0f;
        currentCamera = &orbitalCamera;

        if (scene.cameraExists) {
            fpsCamera = FPSCamera{app, width, height};
            fpsCamera.position = scene.cameraTranslation;
            glm::vec3 eulerAngles = glm::eulerAngles(scene.cameraRotation);

            fpsCamera.pitch = -glm::degrees(eulerAngles.x);
            if (glm::degrees(eulerAngles.x) < -90.0f || 90.0f < glm::degrees(eulerAngles.x)) {
                fpsCamera.pitch = -glm::degrees(eulerAngles.x) + 180;
            }
            fpsCamera.yaw = glm::mod(glm::degrees(eulerAngles.y), 360.0f);
            fpsCamera.fovY = scene.cameraYFov;
            currentCamera = &fpsCamera;
        }
    }

    void createPipelines(const Context& context) {
        std::vector<ShaderHandle> shaders(4);
        shaders[0] = context.createShader({
            .code = readShader("base.rgen", "main"),
            .stage = vk::ShaderStageFlagBits::eRaygenKHR,
        });
        shaders[1] = context.createShader({
            .code = readShader("base.rmiss", "main"),
            .stage = vk::ShaderStageFlagBits::eMissKHR,
        });
        shaders[2] = context.createShader({
            .code = readShader("shadow.rmiss", "main"),
            .stage = vk::ShaderStageFlagBits::eMissKHR,
        });
        shaders[3] = context.createShader({
            .code = readShader("base.rchit", "main"),
            .stage = vk::ShaderStageFlagBits::eClosestHitKHR,
        });

        bloomPass = BloomPass(context, width, height);
        compositePass = CompositePass(context, baseImage, bloomPass.bloomImage, width, height);

        descSet = context.createDescriptorSet({
            .shaders = shaders,
            .buffers =
                {
                    {"VertexBuffers", scene.vertexBuffers},
                    {"IndexBuffers", scene.indexBuffers},
                    {"AddressBuffer", scene.addressBuffer},
                    {"MaterialIndexBuffer", scene.materialIndexBuffer},
                    {"NormalMatrixBuffer", scene.normalMatrixBuffer},
                },
            .images =
                {
                    {"baseImage", baseImage},
                    {"bloomImage", bloomPass.bloomImage},
                },
            .accels = {{"topLevelAS", scene.topAccel}},
        });

        rayTracingPipeline = context.createRayTracingPipeline({
            .rgenShaders = shaders[0],
            .missShaders = {shaders, 1, 2},
            .chitShaders = shaders[3],
            .descSetLayout = descSet->getLayout(),
            .pushSize = sizeof(PushConstants),
            .maxRayRecursionDepth = maxRayRecursionDepth,
        });
    }

    void update() {
        RV_ASSERT(currentCamera, "currentCamera is nullptr");
        currentCamera->processInput();
        pushConstants.frame++;
        pushConstants.invView = currentCamera->getInvView();
        pushConstants.invProj = currentCamera->getInvProj();
    }

    void render(const CommandBuffer& commandBuffer,
                bool playAnimation,
                bool enableBloom,
                int blurIteration) {
        // Update
        if (!playAnimation) {
            return;
        }
        scene.updateTopAccel(commandBuffer.commandBuffer, pushConstants.frame);

        // Ray tracing
        commandBuffer.bindDescriptorSet(descSet, rayTracingPipeline);
        commandBuffer.bindPipeline(rayTracingPipeline);
        commandBuffer.pushConstants(rayTracingPipeline, &pushConstants);
        commandBuffer.traceRays(rayTracingPipeline, width, height, 1);

        commandBuffer.imageBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                                   vk::PipelineStageFlagBits::eComputeShader, {}, baseImage,
                                   vk::AccessFlagBits::eShaderWrite,
                                   vk::AccessFlagBits::eShaderRead);
        commandBuffer.imageBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                                   vk::PipelineStageFlagBits::eComputeShader, {},
                                   bloomPass.bloomImage, vk::AccessFlagBits::eShaderWrite,
                                   vk::AccessFlagBits::eShaderRead);

        // Blur
        if (enableBloom) {
            for (int i = 0; i < blurIteration; i++) {
                bloomPass.render(commandBuffer, width / 8, height / 8, bloomInfo);
            }
        }

        compositePass.render(commandBuffer, width / 8, height / 8, compositeInfo);
    }

    uint32_t width;
    uint32_t height;
    uint32_t maxRayRecursionDepth;

    Scene scene;

    CompositeInfo compositeInfo;
    CompositePass compositePass;
    BloomInfo bloomInfo;
    BloomPass bloomPass;

    ImageHandle baseImage;

    DescriptorSetHandle descSet;
    RayTracingPipelineHandle rayTracingPipeline;

    Camera* currentCamera = nullptr;
    FPSCamera fpsCamera;
    OrbitalCamera orbitalCamera;

    PushConstants pushConstants;
};

class DebugApp : public App {
public:
    DebugApp()
        : App({
              .width = 1920,
              .height = 1080,
              .title = "rtcamp9",
              .layers = Layer::Validation,
              .extensions = Extension::RayTracing,
          }) {
        spdlog::info("Executable directory: {}", getExecutableDirectory().string());
        spdlog::info("Shader source directory: {}", getShaderSourceDirectory().string());
        spdlog::info("SPIR-V directory: {}", getSpvDirectory().string());
        fs::create_directory(getSpvDirectory());

        if (shouldRecompile("base.rgen", "main")) {
            compileShader("base.rgen", "main");
        }
        if (shouldRecompile("base.rchit", "main")) {
            compileShader("base.rchit", "main");
        }
        if (shouldRecompile("base.rmiss", "main")) {
            compileShader("base.rmiss", "main");
        }
        if (shouldRecompile("shadow.rmiss", "main")) {
            compileShader("shadow.rmiss", "main");
        }
        if (shouldRecompile("blur.comp", "main")) {
            compileShader("blur.comp", "main");
        }
        if (shouldRecompile("composite.comp", "main")) {
            compileShader("composite.comp", "main");
        }

        renderer = std::make_unique<Renderer>(context, width, height, this);
    }

    void onUpdate() override { renderer->update(); }

    void drawGUI() {
        auto& pushConstants = renderer->pushConstants;
        auto& compositeInfo = renderer->compositeInfo;
        auto& bloomInfo = renderer->bloomInfo;

        ImGui::SliderInt("Sample count", &pushConstants.sampleCount, 1, 512);

        // Dome light
        ImGui::SliderFloat("Dome light phi", &pushConstants.domeLightPhi, 0.0, 360.0);

        // Infinite light
        ImGui::SliderFloat4("Infinite light direction", &pushConstants.infiniteLightDirection[0],
                            -1.0, 1.0);
        ImGui::SliderFloat("Infinite light intensity", &pushConstants.infiniteLightIntensity, 0.0f,
                           1.0f);

        // Bloom
        ImGui::Checkbox("Enable bloom", &enableBloom);
        if (enableBloom) {
            ImGui::SliderFloat("Bloom intensity", &compositeInfo.bloomIntensity, 0.0, 10.0);
            ImGui::SliderFloat("Bloom threshold", &pushConstants.bloomThreshold, 0.0, 10.0);
            ImGui::SliderInt("Blur iteration", &blurIteration, 0, 64);
            ImGui::SliderInt("Blur size", &bloomInfo.blurSize, 0, 64);
        }

        // Tone mapping
        ImGui::Checkbox("Enable tone mapping",
                        reinterpret_cast<bool*>(&compositeInfo.enableToneMapping));
        if (compositeInfo.enableToneMapping) {
            ImGui::SliderFloat("Exposure", &compositeInfo.exposure, 0.0, 5.0);
        }

        // Gamma correction
        ImGui::Checkbox("Enable gamma correction",
                        reinterpret_cast<bool*>(&compositeInfo.enableGammaCorrection));
        if (compositeInfo.enableGammaCorrection) {
            ImGui::SliderFloat("Gamma", &compositeInfo.gamma, 0.0, 5.0);
        }

        ImGui::Checkbox("Play animation", &playAnimation);
    }

    void onRender(const CommandBuffer& commandBuffer) override {
        drawGUI();

        renderer->render(commandBuffer, playAnimation, enableBloom, blurIteration);

        // Copy to swapchain image
        commandBuffer.copyImage(renderer->compositePass.finalImageBGRA, getCurrentColorImage(),
                                vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR);
    }

    std::unique_ptr<Renderer> renderer;

    // GUI states
    int blurIteration = 32;
    bool enableBloom = false;
    bool playAnimation = true;
};

int main() {
    try {
        DebugApp app{};
        app.run();
    } catch (const std::exception& e) {
        spdlog::error(e.what());
    }
}
