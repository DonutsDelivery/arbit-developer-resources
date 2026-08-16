#include "../src/depth_cache.h"
#include "../src/depth_texture_state.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using videohelper::DepthCacheBinding;

namespace
{
int failures = 0;
void check (bool value, const char* message) { if (!value) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); } }

const std::vector<uint8_t> pngA = {0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,0x49,0x48,0x44,0x52,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x02,0x10,0x00,0x00,0x00,0x00,0x07,0x4d,0x8e,0xbb,0x00,0x00,0x00,0x12,0x49,0x44,0x41,0x54,0x78,0x9c,0x63,0x60,0x60,0x60,0x60,0x64,0x68,0x60,0xf8,0xff,0x1f,0x00,0x05,0x0d,0x02,0x80,0x01,0x70,0xca,0x8c,0x00,0x00,0x00,0x00,0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82};
const std::vector<uint8_t> pngB = {0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,0x49,0x48,0x44,0x52,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x02,0x10,0x00,0x00,0x00,0x00,0x07,0x4d,0x8e,0xbb,0x00,0x00,0x00,0x12,0x49,0x44,0x41,0x54,0x78,0x9c,0x63,0xf8,0xff,0xbf,0x81,0x81,0x81,0x81,0x91,0x81,0x01,0x00,0x14,0x7c,0x02,0x80,0x6d,0x88,0x7b,0x35,0x00,0x00,0x00,0x00,0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82};

void writeBytes (const fs::path& path, const std::vector<uint8_t>& bytes)
{
    std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())); out.close();
}
std::string hashBytes (const std::vector<uint8_t>& bytes) { videohelper::Sha256 h; h.update(bytes.data(), bytes.size()); return h.finishHex(); }

struct Fixture
{
    fs::path root, dir;
    DepthCacheBinding binding;
    Fixture (std::string suffix = "base")
    {
        root = fs::canonical(fs::temp_directory_path()) / ("depth-cache-test-" + suffix + "-" + std::to_string(::getpid()));
        fs::remove_all(root); fs::create_directories(root);
        binding = {"depth-job-sealed/published", "", "depth-sequence-v1", "depth_", ".png", 1, 6, 2, 2, 2, 2.0};
        dir = root / binding.key; fs::create_directories(dir);
        writeBytes(dir / "depth_000001.png", pngA); writeBytes(dir / "depth_000002.png", pngB);
        std::string material = "depth-receipt-v1";
        std::ostringstream fps; fps << std::setprecision(17) << binding.fps;
        for (const auto& v : std::vector<std::string>{binding.version,binding.prefix,binding.extension,"1","6","2","2","2",fps.str()}) material += videohelper::depthcache::field(v);
        for (const auto& item : std::vector<std::pair<std::string,const std::vector<uint8_t>*>>{{"depth_000001.png",&pngA},{"depth_000002.png",&pngB}})
            material += videohelper::depthcache::field(item.first) + videohelper::depthcache::field(std::to_string(item.second->size())) + videohelper::depthcache::field(hashBytes(*item.second));
        binding.receipt = videohelper::sha256Text(material);
        nlohmann::json manifest = {{"schema",3},{"version",binding.version},{"framePrefix",binding.prefix},{"frameExtension",binding.extension},
            {"firstFrame",1},{"frameDigits",6},{"width",2},{"height",2},{"frameCount",2},{"frames",2},{"fps","2"},
            {"contentReceipt",binding.receipt},{"outputSemantics",{{"publication","per-frame min/max normalized uint16 grayscale PNG; constant/non-finite frames are rejected"},{"channel","red/single-channel"}}}};
        std::ofstream(dir / "receipt.json") << manifest.dump() << '\n';
        std::ofstream(dir / ".depth-output-owned") << "depth-output-v1\n";
        seal();
    }
    ~Fixture() { fs::permissions(dir, fs::perms::owner_all, fs::perm_options::add); fs::remove_all(root); }
    void seal()
    {
        for (const auto& entry : fs::directory_iterator(dir)) fs::permissions(entry.path(), fs::perms::owner_read | fs::perms::group_read, fs::perm_options::replace);
    }
};

fs::path replacement;
void replaceAtBarrier (int directoryFd, const char* name) { ::renameat(AT_FDCWD, replacement.c_str(), directoryFd, name); }

bool rejected (Fixture& fixture, double pts = 0.0)
{
    std::string error; return !videohelper::admitDepthFrame(fixture.root.string(), fixture.binding, pts, error);
}

struct FakeRenderer
{
    unsigned next = 40;
    int uploads = 0;
    std::vector<unsigned> deleted;
    unsigned uploadR16 (const uint16_t*, int, int, unsigned)
    {
        ++uploads;
        return ++next;
    }
    void deleteTexture (unsigned texture)
    {
        if (texture != 0) deleted.push_back(texture);
    }
};
}

int main()
{
    {
        Fixture f("valid"); std::string error;
        auto first = videohelper::admitDepthFrame(f.root.string(), f.binding, 0.0, error);
        check(first && first->index() == 0 && first->pixels().size() == 4, "valid PIL I;16 PNG is decoded into owning R16 frame");
        check(first && first->pixels() == std::vector<uint16_t>({0,1,32768,65535}), "host uint16 pixels preserve I;16 values");
        check(first && first->normalized(0) == 0.0f && first->normalized(3) == 1.0f && std::isfinite(first->normalized(2)), "normalization is finite and bounded [0,1]");
        auto exact = videohelper::admitDepthFrame(f.root.string(), f.binding, 0.5, error);
        auto low = videohelper::admitDepthFrame(f.root.string(), f.binding, -100.0, error);
        auto high = videohelper::admitDepthFrame(f.root.string(), f.binding, 100.0, error);
        check(exact && exact->index() == 1 && low && low->index() == 0 && high && high->index() == 1, "PTS resolves exact floor index with endpoint clamps");
        check(videohelper::depthFrameIndexForPts(std::numeric_limits<double>::infinity(),2,2) == -1, "non-finite PTS is rejected");
    }
    { Fixture f("forged"); f.binding.receipt[0] = f.binding.receipt[0] == 'a' ? 'b' : 'a'; check(rejected(f), "forged receipt/hash is rejected"); }
    {
        Fixture f("symlink-dir"); fs::path real=f.dir.string()+"-real"; fs::rename(f.dir,real); fs::create_directory_symlink(real,f.dir); check(rejected(f), "symlink receipt directory is rejected"); fs::remove(f.dir); fs::rename(real,f.dir);
    }
    {
        Fixture f("symlink-frame"); fs::permissions(f.dir/"depth_000001.png",fs::perms::owner_write,fs::perm_options::add); fs::remove(f.dir/"depth_000001.png"); fs::create_symlink(f.dir/"depth_000002.png",f.dir/"depth_000001.png"); check(rejected(f), "symlink frame is rejected");
    }
    {
        Fixture f("hardlink"); fs::path outside=f.root/"outside.png"; fs::create_hard_link(f.dir/"depth_000001.png",outside); check(rejected(f), "hardlinked frame is rejected"); fs::remove(outside);
    }
    {
        Fixture f("replace"); replacement=f.root/"replacement.png"; writeBytes(replacement,pngB); fs::permissions(replacement,fs::perms::owner_read|fs::perms::group_read,fs::perm_options::replace);
        std::string error; check(!videohelper::admitDepthFrame(f.root.string(),f.binding,0.0,error,replaceAtBarrier), "replacement during selected-open barrier is rejected");
    }
    {
        Fixture f("truncate"); fs::permissions(f.dir/"depth_000001.png",fs::perms::owner_write,fs::perm_options::add); fs::resize_file(f.dir/"depth_000001.png",10); f.seal(); check(rejected(f), "truncated frame is rejected");
    }
    {
        Fixture f("oversize"); fs::permissions(f.dir/"depth_000001.png",fs::perms::owner_write,fs::perm_options::add); fs::resize_file(f.dir/"depth_000001.png",videohelper::depthcache::kFrameLimit+1); f.seal(); check(rejected(f), "oversized frame is rejected");
    }
    { Fixture f("extra"); std::ofstream(f.dir/"extra") << "x"; fs::permissions(f.dir/"extra",fs::perms::owner_read,fs::perm_options::replace); check(rejected(f), "extra receipt entry is rejected"); }
    {
        Fixture f("malformed"); fs::permissions(f.dir/"depth_000001.png",fs::perms::owner_write,fs::perm_options::add); writeBytes(f.dir/"depth_000001.png",{1,2,3}); f.seal(); check(rejected(f), "malformed PNG is rejected");
    }
    {
        Fixture f("dimensions"); f.binding.width=3; check(rejected(f), "manifest/decoded dimension mismatch is rejected");
    }
    {
        Fixture first("generation-old");
        FakeRenderer renderer;
        videohelper::DepthTextureState state;
        state.beginHelperGeneration(renderer, 7);
        std::string error;
        auto admitted = videohelper::admitDepthFrame(first.root.string(), first.binding, 0.0, error);
        check(static_cast<bool>(admitted), "initial helper generation securely admits depth receipt before upload");
        if (admitted)
        {
            state.texture = renderer.uploadR16(admitted->pixels().data(), admitted->width(), admitted->height(), 0);
            state.width = admitted->width(); state.height = admitted->height();
            state.frameIndex = admitted->index(); state.receipt = first.binding.receipt;
        }
        const unsigned oldTexture = state.texture;

        state.beginHelperGeneration(renderer, 8);
        check(state.texture == 0 && state.receipt.empty()
                && renderer.deleted == std::vector<unsigned>{oldTexture},
              "helper generation restart deterministically deletes and invalidates the old admitted texture");

        fs::permissions(first.dir/"depth_000001.png",fs::perms::owner_write,fs::perm_options::add);
        writeBytes(first.dir/"depth_000001.png",pngB); first.seal();
        error.clear();
        auto stale = videohelper::admitDepthFrame(first.root.string(), first.binding, 0.0, error);
        check(!stale && renderer.uploads == 1 && state.texture == 0,
              "stale receipt fails closed after restart and cannot upload or render an old handle");

        Fixture fresh("generation-fresh");
        error.clear();
        auto readmitted = videohelper::admitDepthFrame(fresh.root.string(), fresh.binding, 0.0, error);
        check(static_cast<bool>(readmitted), "replacement generation requires a fresh secure receipt admission");
        if (readmitted)
        {
            state.texture = renderer.uploadR16(readmitted->pixels().data(), readmitted->width(), readmitted->height(), 0);
            state.receipt = fresh.binding.receipt;
        }
        const unsigned replacementTexture = state.texture;
        state.invalidateReplacedReceipt(renderer, std::string(64, 'a'));
        check(state.texture == 0 && renderer.deleted.size() == 2
                && renderer.deleted.back() == replacementTexture,
              "receipt replacement deterministically deletes the previously admitted texture before readmission");
    }
    std::fprintf(stderr, failures ? "%d depth cache checks failed\n" : "depth cache admission checks passed\n", failures);
    return failures == 0 ? 0 : 1;
}
