#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

namespace videohelper
{
class CompositorOwnershipGate
{
public:
    enum class Owner : uint8_t { none, exportJob, renderCache, frameProbe };

    class Lease
    {
    public:
        Lease (CompositorOwnershipGate& gate, Owner owner) : gate_ (&gate), owner_ (owner) {}
        ~Lease() { release(); }
        Lease (const Lease&) = delete;
        Lease& operator= (const Lease&) = delete;
        Lease (Lease&& other) noexcept : gate_ (other.gate_), owner_ (other.owner_)
        { other.gate_ = nullptr; }
        void release()
        {
            if (gate_ != nullptr)
            {
                gate_->owner_.store (Owner::none, std::memory_order_release);
                gate_ = nullptr;
            }
        }
    private:
        CompositorOwnershipGate* gate_;
        Owner owner_;
    };

    std::shared_ptr<Lease> tryClaim (Owner requested)
    {
        return tryClaimWithFactory (requested, [] (CompositorOwnershipGate& gate, Owner owner)
        {
            return std::make_shared<Lease> (gate, owner);
        });
    }

    template <typename LeaseFactory>
    std::shared_ptr<Lease> tryClaimWithFactory (Owner requested, LeaseFactory&& makeLease)
    {
        Owner expected = Owner::none;
        if (! owner_.compare_exchange_strong (expected, requested,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire))
            return {};
        try
        {
            return makeLease (*this, requested);
        }
        catch (...)
        {
            // No lease exists to release the claim when allocation fails.
            owner_.store (Owner::none, std::memory_order_release);
            throw;
        }
    }

    Owner owner() const { return owner_.load (std::memory_order_acquire); }

private:
    std::atomic<Owner> owner_ { Owner::none };
};
} // namespace videohelper
