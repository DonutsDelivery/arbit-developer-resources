// block_c_packer.h — Block C voice allocator + texture packer (pre-work A2)
//
// Turns the note/link timeline into the two RGBA32F textures the shader contract
// (§4.3) reads: uNotes (128 rows x 4 texels) and uLinks (256 rows x 1 texel).
//
// The differentiator vs. an FFT visualiser is that row assignment is a STATEFUL
// VOICE ALLOCATOR, not a per-frame sort: a note gets a row when it enters the
// past/future timeline window and keeps that row while it remains selected;
// rows free on exit and are reused. masterRow and the uLinks
// row indices therefore stay valid for a note's entire residency, so age- and
// row-keyed shader effects don't jump when an unrelated note enters or leaves.
//
// The shader-workbench's packScoreTextures() re-rows every frame (contiguous,
// stateless) — fine for a single still, wrong for motion. This packer is the
// real contract: it is what the helper must implement, and the row-stability
// unit tests pin the behaviour.
//
// Dependency-free C++17; reuses the arbitmod note/link model (A1) so plugin and
// helper share ONE definition. Destined for video-helper/src alongside
// mod_defs.h. No GPL code.
//
// uNoteCount contract note: with a sticky allocator, freeing a middle row leaves
// a hole, so uNoteCount is the *upper bound* (highest occupied row + 1), NOT the
// count of active notes. Shaders loop `i < uNoteCount` and read row i; holes are
// written zeroed (velocity 0) and gate to nothing. (This is the one place the
// stable allocator diverges from the workbench's contiguous packer — pinned by
// the noteCountCoversHoles test.)

#pragma once

#include "mod_defs.h"

#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace arbitblockc
{

using arbitmod::Link;
using arbitmod::Note;
using arbitmod::Score;

constexpr int kMaxNotes  = 128;
constexpr int kMaxLinks  = 256;
constexpr int kTexelsPerNote = 4;

struct PackResult
{
    // 128 * 4 texels * 4 channels, row-major (row, texel, channel).
    std::vector<float> notesTex;
    // 256 * 4 channels (one texel per link edge).
    std::vector<float> linksTex;
    int noteCount = 0;   // highest occupied row + 1 (loop bound, see header note)
    int linkCount = 0;   // edges with both endpoints resident
};

class BlockCPacker
{
public:
    BlockCPacker() { reset(); }

    void reset()
    {
        rowToNote_.fill (-1);
        noteToRow_.clear();
    }

    // Priority: sounding > upcoming > recently ended. Selection applies this
    // before row assignment so lower-priority history cannot starve new notes.
    static int residency (const Note& n, float beat,
                          float historyBeats, float lookaheadBeats)
    {
        return arbitmod::timelineResidency (n, beat, historyBeats, lookaheadBeats);
    }

    int rowOf (int noteId) const
    {
        auto it = noteToRow_.find (noteId);
        return it == noteToRow_.end() ? -1 : it->second;
    }

    PackResult pack (const Score& score, float beat)
    {
        return pack (score, beat, score.historyBeats, score.lookaheadBeats);
    }

    PackResult pack (const Score& score, float beat,
                     float historyBeats, float lookaheadBeats)
    {
        // 1. Select the best bounded resident set. Within a class, sounding and
        // upcoming notes sort by onset; history sorts newest end first.
        struct Candidate { const Note* note; int priority; };
        std::vector<Candidate> candidates;
        candidates.reserve (score.notes.size());
        for (const auto& note : score.notes)
        {
            if (note.muted)
                continue;
            const int priority = residency (note, beat, historyBeats, lookaheadBeats);
            if (priority > 0)
                candidates.push_back ({ &note, priority });
        }
        std::sort (candidates.begin(), candidates.end(),
                   [] (const Candidate& a, const Candidate& b)
                   {
                       if (a.priority != b.priority) return a.priority > b.priority;
                       if (a.priority == 1 && a.note->endBeat() != b.note->endBeat())
                           return a.note->endBeat() > b.note->endBeat();
                       if (a.note->startBeat != b.note->startBeat)
                           return a.note->startBeat < b.note->startBeat;
                       return a.note->id < b.note->id;
                   });
        if (candidates.size() > static_cast<size_t> (kMaxNotes))
            candidates.resize (kMaxNotes);

        std::unordered_set<int> selectedIds;
        selectedIds.reserve (candidates.size());
        for (const auto& candidate : candidates)
            selectedIds.insert (candidate.note->id);

        // 2. Free rows outside the selected set. This is the capacity-aware
        // eviction step: active/upcoming entrants can displace older history.
        for (int row = 0; row < kMaxNotes; ++row)
        {
            const int id = rowToNote_[row];
            if (id < 0) continue;
            if (selectedIds.find (id) == selectedIds.end())
            {
                rowToNote_[row] = -1;
                noteToRow_.erase (id);
            }
        }

        // 3. Assign the selected entrants to the lowest free rows. Notes that
        // survived selection keep their old rows, preserving motion continuity.
        for (const auto& candidate : candidates)
        {
            if (noteToRow_.find (candidate.note->id) != noteToRow_.end())
                continue;
            int freeRow = -1;
            for (int row = 0; row < kMaxNotes; ++row)
                if (rowToNote_[row] < 0) { freeRow = row; break; }
            if (freeRow < 0) break;
            rowToNote_[freeRow] = candidate.note->id;
            noteToRow_[candidate.note->id] = freeRow;
        }

        // 4. Emit textures.
        PackResult out;
        out.notesTex.assign (static_cast<size_t> (kMaxNotes) * kTexelsPerNote * 4u, 0.0f);
        out.linksTex.assign (static_cast<size_t> (kMaxLinks) * 4u, 0.0f);

        int maxRow = -1;
        for (int row = 0; row < kMaxNotes; ++row)
        {
            const int id = rowToNote_[row];
            if (id < 0) continue;
            const Note* n = score.noteById (id);
            if (n == nullptr) continue;
            maxRow = row;
            writeNote (out.notesTex, row, *n, score, beat);
        }
        out.noteCount = maxRow + 1;

        // 5. Link edges: both endpoints must be resident.
        for (const auto& l : score.links)
        {
            const int sRow = rowOf (l.slaveNoteId);
            const int mRow = rowOf (l.masterNoteId);
            if (sRow < 0 || mRow < 0) continue;
            if (out.linkCount >= kMaxLinks) break;
            const size_t off = static_cast<size_t> (out.linkCount++) * 4u;
            out.linksTex[off + 0] = static_cast<float> (sRow);
            out.linksTex[off + 1] = static_cast<float> (mRow);
            out.linksTex[off + 2] = static_cast<float> (l.slaveHarmonic);   // num
            out.linksTex[off + 3] = static_cast<float> (l.masterHarmonic);  // den
        }
        return out;
    }

private:
    void set (std::vector<float>& tex, int row, int texel, int ch, float v) const
    {
        tex[(static_cast<size_t> (row) * kTexelsPerNote + static_cast<size_t> (texel)) * 4u
            + static_cast<size_t> (ch)] = v;
    }

    void writeNote (std::vector<float>& tex, int row, const Note& n,
                    const Score& score, float beat) const
    {
        const float vel   = arbitmod::clamp01 (n.velocity / 127.0f);
        const float age   = beat - n.startBeat;        // negative for lookahead notes
        const float remain = n.endBeat() - beat;
        const float bentFrequency = arbitmod::pitchFrequencyAtBeat (n, beat);
        const float bendSemitones = n.freqHz > 0.0f && bentFrequency > 0.0f
            ? 12.0f * std::log2 (bentFrequency / n.freqHz) : 0.0f;
        const float cents = arbitmod::centsFromRoot (bentFrequency, score.rootFreq);
        const int   masterRow = (n.linkMasterId >= 0) ? rowOf (n.linkMasterId) : -1;

        // texel0: midiNote, velocity/127, ageBeats, remainBeats
        set (tex, row, 0, 0, n.midiNote);
        set (tex, row, 0, 1, vel);
        set (tex, row, 0, 2, age);
        set (tex, row, 0, 3, remain);
        // texel1: frame-resolved freqHz, centsFromRoot, trackId, isRoot
        set (tex, row, 1, 0, bentFrequency);
        set (tex, row, 1, 1, cents);
        set (tex, row, 1, 2, static_cast<float> (n.trackId));
        set (tex, row, 1, 3, n.isRoot ? 1.0f : 0.0f);
        // texel2: e2,e3,e5,e7
        set (tex, row, 2, 0, n.primes[0]);
        set (tex, row, 2, 1, n.primes[1]);
        set (tex, row, 2, 2, n.primes[2]);
        set (tex, row, 2, 3, n.primes[3]);
        // texel3: e11,e13, masterRow, pitch bend in semitones
        set (tex, row, 3, 0, n.primes[4]);
        set (tex, row, 3, 1, n.primes[5]);
        set (tex, row, 3, 2, static_cast<float> (masterRow));
        set (tex, row, 3, 3, bendSemitones);
    }

    std::array<int, kMaxNotes> rowToNote_ {};
    std::unordered_map<int, int> noteToRow_;
};

// Convenience accessors for tests / consumers reading PackResult texels.
inline float noteTexel (const PackResult& r, int row, int texel, int ch)
{
    return r.notesTex[(static_cast<size_t> (row) * kTexelsPerNote + static_cast<size_t> (texel)) * 4u
                      + static_cast<size_t> (ch)];
}
inline float linkTexel (const PackResult& r, int edge, int ch)
{
    return r.linksTex[static_cast<size_t> (edge) * 4u + static_cast<size_t> (ch)];
}

} // namespace arbitblockc
