#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>

// SDFormat parsing
#include <sdf/sdf.hh>
#include <sdf/Mesh.hh>

// Point Cloud Library
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

// Assimp for mesh loading
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// For thread pool
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

// Use gz math headers instead of the deprecated ignition versions
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Color.hh>

// -----------------------------------------------------------------------------
// ThreadPool for parallel processing
class ThreadPool {
private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;

public:
  ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i) {
      workers.emplace_back([this]{
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty())
              return;
            task = std::move(tasks.front());
            tasks.pop();
          }
          task();
        }
      });
    }
  }

  template<class F>
  void enqueue(F&& f) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks.emplace(std::forward<F>(f));
    }
    condition.notify_one();
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
      worker.join();
  }
};

// -----------------------------------------------------------------------------
// SDFToPointCloud class definition
//
// This version accesses a visual’s geometry via its XML element using visual->Element().
// It then checks which geometry element exists (mesh, sphere, box, cylinder).
class SDFToPointCloud {
private:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud_;
  std::mutex cloud_mutex_;
  ThreadPool thread_pool_;
  double leaf_size_;

public:
  SDFToPointCloud(size_t num_threads = 4, double leaf_size = 0.01)
      : combined_cloud_(new pcl::PointCloud<pcl::PointXYZRGB>),
        thread_pool_(num_threads),
        leaf_size_(leaf_size) {
    // Set URI resolution callback if needed.
    sdf::setFindCallback([](const std::string &_uri) { return _uri; });
  }

  bool loadSDF(const std::string &sdf_file) {
    sdf::Root root;
    sdf::Errors errors = root.Load(sdf_file);
    if (!errors.empty()) {
      std::cerr << "SDF parsing errors:" << std::endl;
      for (const auto &error : errors)
        std::cerr << error << std::endl;
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      combined_cloud_->clear();
    }
    // Traverse worlds using the legacy API
    for (uint64_t w = 0; w < root.WorldCount(); ++w) {
      const sdf::World *world = root.WorldByIndex(w);
      for (uint64_t m = 0; m < world->ModelCount(); ++m) {
        const sdf::Model *model = world->ModelByIndex(m);
        processModel(model, gz::math::Pose3d::Zero);
      }
    }
    return true;
  }

  // Process model: iterate links and visuals.
  void processModel(const sdf::Model *model, const gz::math::Pose3d &parent_pose) {
    gz::math::Pose3d model_pose = parent_pose * model->RawPose();
    for (uint64_t l = 0; l < model->LinkCount(); ++l) {
      const sdf::Link *link = model->LinkByIndex(l);
      gz::math::Pose3d link_pose = model_pose * link->RawPose();
      for (uint64_t v = 0; v < link->VisualCount(); ++v) {
        const sdf::Visual *visual = link->VisualByIndex(v);
        gz::math::Pose3d visual_pose = link_pose * visual->RawPose();

        // Get the XML element for the visual.
        sdf::ElementPtr visualElem = visual->Element();
        if (!visualElem->HasElement("geometry"))
          continue;
        sdf::ElementPtr geomElem = visualElem->GetElement("geometry");
        // Instead of reading a "type" attribute, determine the type by checking for known elements.
        std::string geomType;
        if (geomElem->HasElement("mesh"))
          geomType = "mesh";
        else if (geomElem->HasElement("sphere"))
          geomType = "sphere";
        else if (geomElem->HasElement("box"))
          geomType = "box";
        else if (geomElem->HasElement("cylinder"))
          geomType = "cylinder";
        else
          geomType = "";

        if (geomType == "mesh") {
          sdf::ElementPtr meshElem = geomElem->GetElement("mesh");
          std::string meshUri = meshElem->Get<std::string>("uri");
          // Default scale is 1,1,1 if not defined.
          gz::math::Vector3d scale = meshElem->HasElement("scale") ?
              meshElem->Get<gz::math::Vector3d>("scale") : gz::math::Vector3d::One;
          thread_pool_.enqueue([this, meshUri, visual_pose, scale]() {
            processMesh(meshUri, visual_pose, scale);
          });
        } else {
          thread_pool_.enqueue([this, visual]() {
            processGeometry(visual);
          });
        }
      }
    }
    // Process nested models recursively.
    for (uint64_t nm = 0; nm < model->ModelCount(); ++nm) {
      const sdf::Model *nested_model = model->ModelByIndex(nm);
      processModel(nested_model, model_pose);
    }
  }

  void processMesh(const std::string &mesh_uri,
                   const gz::math::Pose3d &pose,
                   const gz::math::Vector3d &scale) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(mesh_uri,
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
      std::cerr << "Error loading mesh: " << importer.GetErrorString() << std::endl;
      return;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
      aiMesh* mesh = scene->mMeshes[m];
      for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
        const aiFace& face = mesh->mFaces[f];
        if (face.mNumIndices != 3)
          continue; // Only process triangles.
        for (unsigned int i = 0; i < 3; ++i) {
          aiVector3D vertex = mesh->mVertices[face.mIndices[i]];
          vertex.x *= scale.X();
          vertex.y *= scale.Y();
          vertex.z *= scale.Z();
          pcl::PointXYZRGB point;
          point.x = vertex.x;
          point.y = vertex.y;
          point.z = vertex.z;
          if (mesh->HasVertexColors(0)) {
            aiColor4D color = mesh->mColors[0][face.mIndices[i]];
            point.r = static_cast<uint8_t>(color.r * 255);
            point.g = static_cast<uint8_t>(color.g * 255);
            point.b = static_cast<uint8_t>(color.b * 255);
          } else if (scene->mMaterials && mesh->mMaterialIndex < scene->mNumMaterials) {
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
            aiColor4D diffuse;
            if (AI_SUCCESS == material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse)) {
              point.r = static_cast<uint8_t>(diffuse.r * 255);
              point.g = static_cast<uint8_t>(diffuse.g * 255);
              point.b = static_cast<uint8_t>(diffuse.b * 255);
            } else {
              point.r = point.g = point.b = 255;
            }
          } else {
            point.r = point.g = point.b = 255;
          }
          mesh_cloud->points.push_back(point);
        }
      }
    }
    if (mesh_cloud->points.empty())
      return;
    mesh_cloud->width = mesh_cloud->points.size();
    mesh_cloud->height = 1;
    mesh_cloud->is_dense = false;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z();
    Eigen::Quaternionf q(pose.Rot().W(), pose.Rot().X(), pose.Rot().Y(), pose.Rot().Z());
    transform.rotate(q);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*mesh_cloud, *transformed_cloud, transform);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud(transformed_cloud);
    voxel_filter.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    voxel_filter.filter(*downsampled_cloud);

    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      *combined_cloud_ += *downsampled_cloud;
    }
  }

  // Process primitive geometries (box, sphere, cylinder).
  void processGeometry(const sdf::Visual *visual) {
    sdf::ElementPtr visualElem = visual->Element();
    if (!visualElem->HasElement("geometry"))
      return;
    sdf::ElementPtr geomElem = visualElem->GetElement("geometry");
    // Determine geometry type by checking available sub-elements.
    std::string geomType;
    if (geomElem->HasElement("mesh"))
      geomType = "mesh";  // unlikely for this path
    else if (geomElem->HasElement("sphere"))
      geomType = "sphere";
    else if (geomElem->HasElement("box"))
      geomType = "box";
    else if (geomElem->HasElement("cylinder"))
      geomType = "cylinder";
    else
      return;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr geom_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    gz::math::Pose3d pose = visual->RawPose();
    gz::math::Color diffuse_color(1.0, 1.0, 1.0, 1.0);
    if (visual->Material())
      diffuse_color = visual->Material()->Diffuse();
    uint8_t r = static_cast<uint8_t>(diffuse_color.R() * 255);
    uint8_t g = static_cast<uint8_t>(diffuse_color.G() * 255);
    uint8_t b = static_cast<uint8_t>(diffuse_color.B() * 255);

    if (geomType == "box") {
      sdf::ElementPtr boxElem = geomElem->GetElement("box");
      gz::math::Vector3d size = boxElem->Get<gz::math::Vector3d>("size");
      const int samples_per_dim = 10;
      for (int ix = 0; ix < samples_per_dim; ++ix) {
        for (int iy = 0; iy < samples_per_dim; ++iy) {
          for (int iz = 0; iz < samples_per_dim; ++iz) {
            if (ix > 0 && ix < samples_per_dim - 1 &&
                iy > 0 && iy < samples_per_dim - 1 &&
                iz > 0 && iz < samples_per_dim - 1)
              continue;
            float x = (ix / static_cast<float>(samples_per_dim - 1) - 0.5f) * size.X();
            float y = (iy / static_cast<float>(samples_per_dim - 1) - 0.5f) * size.Y();
            float z = (iz / static_cast<float>(samples_per_dim - 1) - 0.5f) * size.Z();
            pcl::PointXYZRGB point;
            point.x = x; point.y = y; point.z = z;
            point.r = r; point.g = g; point.b = b;
            geom_cloud->points.push_back(point);
          }
        }
      }
    } else if (geomType == "sphere") {
      sdf::ElementPtr sphereElem = geomElem->GetElement("sphere");
      double radius = sphereElem->Get<double>("radius");
      const int samples_theta = 20, samples_phi = 10;
      for (int i = 0; i < samples_theta; ++i) {
        for (int j = 0; j < samples_phi; ++j) {
          float theta = i * 2.0f * M_PI / samples_theta;
          float phi = j * M_PI / samples_phi;
          float x = radius * sin(phi) * cos(theta);
          float y = radius * sin(phi) * sin(theta);
          float z = radius * cos(phi);
          pcl::PointXYZRGB point;
          point.x = x; point.y = y; point.z = z;
          point.r = r; point.g = g; point.b = b;
          geom_cloud->points.push_back(point);
        }
      }
    } else if (geomType == "cylinder") {
      sdf::ElementPtr cylinderElem = geomElem->GetElement("cylinder");
      double radius = cylinderElem->Get<double>("radius");
      double length = cylinderElem->Get<double>("length");
      const int samples_height = 10, samples_circle = 20;
      for (int i = 0; i < samples_circle; ++i) {
        for (int j = 0; j < samples_height; ++j) {
          float theta = i * 2.0f * M_PI / samples_circle;
          float h = j * length / (samples_height - 1) - length / 2.0f;
          float x = radius * cos(theta);
          float y = radius * sin(theta);
          float z = h;
          pcl::PointXYZRGB point;
          point.x = x; point.y = y; point.z = z;
          point.r = r; point.g = g; point.b = b;
          geom_cloud->points.push_back(point);
        }
      }
      for (int i = 0; i < samples_circle; ++i) {
        for (int j = 0; j < samples_circle / 2; ++j) {
          float theta = i * 2.0f * M_PI / samples_circle;
          float r_factor = j / static_cast<float>(samples_circle / 2);
          float x = r_factor * radius * cos(theta);
          float y = r_factor * radius * sin(theta);
          pcl::PointXYZRGB point1, point2;
          point1.x = x; point1.y = y; point1.z = -length / 2.0f;
          point2.x = x; point2.y = y; point2.z =  length / 2.0f;
          point1.r = point2.r = r;
          point1.g = point2.g = g;
          point1.b = point2.b = b;
          geom_cloud->points.push_back(point1);
          geom_cloud->points.push_back(point2);
        }
      }
    } else {
      return;
    }

    geom_cloud->width = geom_cloud->points.size();
    geom_cloud->height = 1;
    geom_cloud->is_dense = false;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z();
    Eigen::Quaternionf q(pose.Rot().W(), pose.Rot().X(), pose.Rot().Y(), pose.Rot().Z());
    transform.rotate(q);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*geom_cloud, *transformed_cloud, transform);

    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      *combined_cloud_ += *transformed_cloud;
    }
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloud() {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    return combined_cloud_;
  }

  void savePointCloud(const std::string &filename) {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    pcl::io::savePCDFileBinary(filename, *combined_cloud_);
  }

  void setDownsamplingLeafSize(double leaf_size) {
    leaf_size_ = leaf_size;
  }

};

// // -----------------------------------------------------------------------------
// // Main function
// int main(int argc, char **argv) {
//   if (argc < 2) {
//     std::cerr << "Usage: " << argv[0] << " <sdf_file> [output_pcd] [leaf_size]" << std::endl;
//     return 1;
//   }
//   std::string sdf_file = argv[1];
//   std::string output_pcd = (argc > 2) ? argv[2] : "output.pcd";
//   double leaf_size = (argc > 3) ? std::stod(argv[3]) : 0.01;

//   auto start = std::chrono::high_resolution_clock::now();
//   SDFToPointCloud converter(std::thread::hardware_concurrency(), leaf_size);
//   if (!converter.loadSDF(sdf_file)) {
//     std::cerr << "Failed to load SDF file: " << sdf_file << std::endl;
//     return 1;
//   }
//   auto end = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = end - start;
//   std::cout << "Processing time: " << elapsed.count() << " seconds" << std::endl;

//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = converter.getPointCloud();
//   std::cout << "Generated point cloud with " << cloud->points.size() << " points" << std::endl;

//   converter.savePointCloud(output_pcd);
//   std::cout << "Saved point cloud to " << output_pcd << std::endl;

//   return 0;
// }
