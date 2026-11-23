// run_geant4.cc
//
// GEANT4 driver: generates positron transport raw data in the same text
// schema as the PENELOPE driver (run_penelope.f), so both
// engines' output can be compacted by the shared build_tables.py.
//
// Configuration is read from 'input/sim.in' (step distance, energy
// grid, histories, material folder) -- the same config file the PENELOPE
// driver reads; the path can be overridden with the environment variable
// TABLEGEN_IN. Optional keyword lines after the six positional lines
// (see the "LUT format" section 5 of the README), identical
// to the PENELOPE driver:
//   NSURV_MIN    n   target surviving histories per energy (0 = off)
//   NHIST_MAX    n   hard cap on histories per energy (default NHIST)
//   MAT_LIST     l   material subset, e.g. "3,7,12-14" (default: all)
//   E_INDEX_LIST l   0-based energy-index subset, e.g. "40-45,52"
//   SEED         n   seed base: material N is seeded with SEED + N (default 4000;
//                    the PENELOPE driver uses the same rule)
//   OUT_TAG      s   file-name tag for subset runs (default "extra")
// At each energy NHIST histories are simulated; while the number of
// survivors is below NSURV_MIN and the history count is below NHIST_MAX,
// further batches of NHIST are simulated (the last one clipped to the
// cap). Every history is written in the unchanged raw schema, so n_hist
// per row is simply the number of rows. When MAT_LIST or E_INDEX_LIST is
// present, output goes to output/geant4/raw_extra/matN_<OUT_TAG>.dat
// instead of output/geant4/raw/matN.dat. Without keyword lines the
// behaviour is identical to the previous driver.
// Materials are discovered the same way (MAT_DIR/mat1.mat,
// mat2.mat, ... until one is missing) and read directly from PENELOPE's
// .mat files: only the density + composition header is parsed (mass
// density and atomic-number/atoms-per-molecule lines); the oscillator and
// shell data that follows is PENELOPE-specific physics data and is
// ignored -- GEANT4 derives its own cross sections from density and
// elemental composition. This keeps material definitions a single source
// of truth shared between both engines.
//
// For each material, for each energy, simulates NHIST positrons through a
// path length of STEP_CM in a large block of the material (sized well
// beyond the positron range so geometry boundaries never intervene). Per
// history, records: scattering angle (theta, from the initial +z
// direction), kinetic-energy loss (dE_kin), an approximate total deposited
// energy (DE_total, informational, not used by build_tables.py), and
// whether the positron's kinetic energy reached ~0 within that path length
// ("annihilated") or exited the path length still moving ("survived"), and
// the end position relative to the start (z along the initial direction,
// r perpendicular to it) -- for annihilated histories, where the positron
// stopped inside the step -- and dphi = atan2(x v - y u, x u + y v), the
// azimuth of the final direction's transverse part (u, v) measured from the
// lateral displacement (x, y), 0 when either vanishes (10th column since
// 2026-10-30, the PENELOPE driver's definition). The config line records the
// Geant4 version (engine_version=, from G4Version.hh).
//
// Physics (Oct 2026 fix): G4EmLivermorePhysics by default (PHYSICS OPT4 /
// PHYSICS OPT3 select G4EmStandardPhysics_option4/3); the fixed path length STEP_CM is enforced exactly with a
// G4StepLimiter whose limit is reset to the remaining path after every step;
// tracking cut 50 eV and production-threshold floor 100 eV to mirror the
// PENELOPE driver (EABS = 50 eV, WCC = WCR = 100 eV).
// Previous text (kept for history): G4EmPenelopePhysics (accurate down to ~100 eV, the model Geant4
// recommends for medical/dosimetry work, and the same underlying physics as
// this project's PENELOPE engine). Only the primary positron is ever
// transported -- StackingAction kills every secondary (delta electrons,
// bremsstrahlung/annihilation photons) at production and deposits its
// kinetic energy locally, matching PENELOPE's single-particle-per-history
// treatment. No electron or photon is simulated as a separate track.
//
// Output (per material):
//   output/geant4/raw/matXX.dat  -- per-history raw text, same schema as
//                                    PENELOPE's output/penelope/raw/matXX.dat
//
// This driver does not write the compact binary per-material table that
// the PENELOPE Fortran driver writes inline -- run build_tables.py
// --engine geant4 --from-raw to produce it from the raw text, exactly as
// for PENELOPE's raw output.

#include <cmath>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>   // mkdir() for output/<engine>/raw_extra

#include "G4RunManager.hh"
#include "G4Version.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserEventAction.hh"
#include "G4UserStackingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmStandardPhysics_option3.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4UserLimits.hh"
#include "G4EmParameters.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4NistManager.hh"
#include "G4ParticleGun.hh"
#include "G4Positron.hh"
#include "G4SystemOfUnits.hh"
#include "G4Event.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4TrackStatus.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"

// ---------------------------------------------------------------------------
// Config (shared input/sim.in: six positional lines as for the
// PENELOPE driver, then optional "KEY VALUE" lines)
// ---------------------------------------------------------------------------
struct Config {
    double step_cm = 0.05;
    double emin_ev = 1000.0;
    double emax_ev = 4.6e6;
    int n_energies = 103;
    int n_hist = 100000;
    std::string mat_dir = "input/mats";
    // Optional keywords (defaults reproduce the previous behaviour).
    long nsurv_min = 0;          // 0 = adaptive batches disabled
    long nhist_max = -1;         // -1 = NHIST (resolved after parsing)
    long seed = 4000;            // material N: setTheSeed(seed + N), as in the PENELOPE driver
    std::set<int> mat_list;      // empty = all materials (1-based ids)
    std::set<int> e_index_list;  // empty = all energies (0-based indices)
    std::string out_tag = "extra";
    // Physics / transport options (defaults = the consistency fix).
    std::string physics = "LIV";    // LIV | OPT4 | OPT3  (G4EmLivermorePhysics, G4EmStandardPhysics_option4/3)
    int step_limit = 1;             // 1: end the last step exactly at STEP_CM (G4StepLimiter)
    double lowest_e_ev = 50.0;      // tracking cut for e+/e- [eV] (PENELOPE EABS = 50 eV)
    double cut_floor_ev = 100.0;    // lower edge of the production-threshold table [eV]
    double cut_um = 0.001;          // range cut for secondaries [um]; 1 nm -> every threshold at the 100 eV floor
    double max_step_frac = 1.0;     // extra cap on every step: STEP_CM * frac (convergence tests)
    bool subset() const { return !mat_list.empty() || !e_index_list.empty(); }
};

// Parse "3,7,12-14" (comma/blank separated values and inclusive ranges)
// into a set, requiring lo <= v <= hi. Returns false on any error.
static bool parse_index_list(const std::string &text, int lo, int hi,
                             std::set<int> &out) {
    std::string buf = text;
    for (char &c : buf) if (c == ',' || c == '\t') c = ' ';
    std::istringstream iss(buf);
    std::string tok;
    bool any = false;
    while (iss >> tok) {
        long a = 0, b = 0;
        size_t dash = tok.find('-', 1);  // a leading '-' is a sign, not a range
        try {
            size_t used = 0;
            if (dash != std::string::npos) {
                a = std::stol(tok.substr(0, dash), &used);
                if (used != dash) return false;
                std::string rest = tok.substr(dash + 1);
                b = std::stol(rest, &used);
                if (rest.empty() || used != rest.size()) return false;
            } else {
                a = std::stol(tok, &used);
                if (used != tok.size()) return false;
                b = a;
            }
        } catch (const std::exception &) {
            return false;
        }
        if (a > b || a < lo || b > hi) return false;
        for (long v = a; v <= b; v++) out.insert(static_cast<int>(v));
        any = true;
    }
    return any;
}

static bool parse_long(const std::string &text, long &out) {
    try {
        size_t used = 0;
        out = std::stol(text, &used);
        return used == text.size();
    } catch (const std::exception &) {
        return false;
    }
}

static std::string strip_comment_and_trim(std::string line) {
    auto bang = line.find('!');
    if (bang != std::string::npos) line = line.substr(0, bang);
    size_t a = line.find_first_not_of(" \t\r\n");
    size_t b = line.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return line.substr(a, b - a + 1);
}

static Config read_config(const std::string &path) {
    Config cfg;
    std::ifstream f(path);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        std::exit(1);
    }
    std::vector<std::string> fields;
    std::string line;
    // (fields.size() is tested first so that the line after the sixth
    // field is not consumed before the keyword loop below sees it.)
    while (fields.size() < 6 && std::getline(f, line)) {
        std::string cleaned = strip_comment_and_trim(line);
        if (!cleaned.empty()) fields.push_back(cleaned);
    }
    if (fields.size() < 6) {
        std::cerr << "ERROR: malformed " << path << "\n";
        std::exit(1);
    }
    cfg.step_cm    = std::stod(fields[0]);
    cfg.emin_ev    = std::stod(fields[1]);
    cfg.emax_ev    = std::stod(fields[2]);
    cfg.n_energies = std::stoi(fields[3]);
    cfg.n_hist     = std::stoi(fields[4]);
    cfg.mat_dir    = fields[5];

    // Optional keyword lines: "KEY VALUE [! comment]"; '=' and tabs are
    // treated as blanks, keys are case-insensitive, blank lines ignored.
    while (std::getline(f, line)) {
        std::string cleaned = strip_comment_and_trim(line);
        if (cleaned.empty()) continue;
        for (char &c : cleaned) if (c == '=' || c == '\t') c = ' ';
        std::istringstream iss(cleaned);
        std::string key;
        iss >> key;
        for (char &c : key) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        std::string value;
        std::getline(iss, value);
        value = strip_comment_and_trim(value);
        bool ok = true;
        if (key == "NSURV_MIN") {
            ok = parse_long(value, cfg.nsurv_min) && cfg.nsurv_min >= 0;
        } else if (key == "NHIST_MAX") {
            ok = parse_long(value, cfg.nhist_max) && cfg.nhist_max >= 1;
        } else if (key == "MAT_LIST") {
            cfg.mat_list.clear();
            ok = parse_index_list(value, 1, 200, cfg.mat_list);
        } else if (key == "E_INDEX_LIST") {
            cfg.e_index_list.clear();
            ok = parse_index_list(value, 0, cfg.n_energies - 1, cfg.e_index_list);
        } else if (key == "SEED") {
            ok = parse_long(value, cfg.seed) && cfg.seed >= 1;
        } else if (key == "PHYSICS") {
            for (char &c : value) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            ok = (value == "OPT4" || value == "OPT3" || value == "LIV");
            if (ok) cfg.physics = value;
        } else if (key == "STEP_LIMIT") {
            long v = 0; ok = parse_long(value, v) && (v == 0 || v == 1);
            if (ok) cfg.step_limit = static_cast<int>(v);
        } else if (key == "LOWEST_E_EV") {
            try { cfg.lowest_e_ev = std::stod(value); } catch (...) { ok = false; }
            ok = ok && cfg.lowest_e_ev > 0.0;
        } else if (key == "CUT_FLOOR_EV") {
            try { cfg.cut_floor_ev = std::stod(value); } catch (...) { ok = false; }
            ok = ok && cfg.cut_floor_ev > 0.0;
        } else if (key == "CUT_UM") {
            try { cfg.cut_um = std::stod(value); } catch (...) { ok = false; }
            ok = ok && cfg.cut_um > 0.0;
        } else if (key == "MAX_STEP_FRAC") {
            try { cfg.max_step_frac = std::stod(value); } catch (...) { ok = false; }
            ok = ok && cfg.max_step_frac > 0.0 && cfg.max_step_frac <= 1.0;
        } else if (key == "OUT_TAG") {
            ok = !value.empty() && value.find(' ') == std::string::npos &&
                 value.size() <= 32;
            if (ok) cfg.out_tag = value;
        } else {
            std::cerr << "ERROR: unknown keyword in config: " << key << "\n";
            std::exit(1);
        }
        if (!ok) {
            std::cerr << "ERROR: bad value for " << key << ": " << value << "\n";
            std::exit(1);
        }
    }
    if (cfg.nhist_max < 0) cfg.nhist_max = cfg.n_hist;
    if (cfg.nhist_max < cfg.n_hist) {
        std::cerr << "ERROR: NHIST_MAX must be >= NHIST\n";
        std::exit(1);
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Material translation: parse PENELOPE .mat file header only.
// ---------------------------------------------------------------------------
struct MatSpec {
    double density_g_cm3 = 0.0;
    double mean_excitation_ev = 0.0;   // PENELOPE's I value (0 = not found)
    std::vector<int> z;
    std::vector<double> atoms_per_molecule;
};

static MatSpec parse_penelope_mat(const std::string &path) {
    MatSpec spec;
    std::ifstream f(path);
    if (!f) {
        std::cerr << "ERROR: cannot open material file " << path << "\n";
        std::exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("Mass density") != std::string::npos) {
            auto eq = line.find('=');
            std::istringstream iss(line.substr(eq + 1));
            iss >> spec.density_g_cm3;
        } else if (line.find("Mean excitation energy") != std::string::npos) {
            std::istringstream iss(line.substr(line.find('=') + 1));
            iss >> spec.mean_excitation_ev;
            break;  // oscillator/shell data follows -- PENELOPE-specific, ignored
        } else if (line.find("Number of elements") != std::string::npos) {
            auto eq = line.find('=');
            int n_elements = 0;
            std::istringstream iss(line.substr(eq + 1));
            iss >> n_elements;
            for (int i = 0; i < n_elements; i++) {
                if (!std::getline(f, line)) break;
                // "   atomic number =  7,  atoms/molecule = 1.00000000E+00"
                auto first_eq = line.find('=');
                auto comma = line.find(',', first_eq);
                auto second_eq = line.find('=', comma);
                int zz = 0;
                double aa = 0.0;
                std::istringstream iss_z(line.substr(first_eq + 1, comma - first_eq - 1));
                iss_z >> zz;
                std::istringstream iss_a(line.substr(second_eq + 1));
                iss_a >> aa;
                spec.z.push_back(zz);
                spec.atoms_per_molecule.push_back(aa);
            }
        }
    }
    if (spec.density_g_cm3 <= 0.0 || spec.z.empty() || spec.mean_excitation_ev <= 0.0) {
        std::cerr << "ERROR: failed to parse material header from " << path << "\n";
        std::exit(1);
    }
    return spec;
}

static G4Material *build_g4_material(const MatSpec &spec, int mat_id) {
    auto *nist = G4NistManager::Instance();

    // PENELOPE's atoms/molecule are a pseudo-stoichiometric composition
    // (may be fractional for compounds/mixtures), not a real molecular
    // formula. Convert to mass fractions -- valid for any composition --
    // rather than AddElementByNumberOfAtoms, which requires integer counts.
    std::vector<G4Element *> elements;
    std::vector<double> masses;
    double total_mass = 0.0;
    for (size_t i = 0; i < spec.z.size(); i++) {
        G4Element *elm = nist->FindOrBuildElement(spec.z[i]);
        elements.push_back(elm);
        double m = spec.atoms_per_molecule[i] * (elm->GetA() / (g / mole));
        masses.push_back(m);
        total_mass += m;
    }

    std::string name = "Schneider_mat" + std::to_string(mat_id);
    auto *material = new G4Material(name, spec.density_g_cm3 * g / cm3,
                                     (G4int)elements.size());
    for (size_t i = 0; i < elements.size(); i++) {
        material->AddElement(elements[i], masses[i] / total_mass);
    }
    // Use PENELOPE's mean excitation energy instead of Geant4's own
    // (Bragg-additivity) value, so both engines share the same stopping power.
    material->GetIonisation()->SetMeanExcitationEnergy(spec.mean_excitation_ev * eV);
    return material;
}

// Print the energy thresholds the range cut turned into for this material
// (the cuts table is only built at the first BeamOn after a change), and
// the mean excitation energy actually in use.
static void print_cuts(int mat_id) {
    auto *table = G4ProductionCutsTable::GetProductionCutsTable();
    for (size_t i = 0; i < table->GetTableSize(); i++) {
        const G4MaterialCutsCouple *couple = table->GetMaterialCutsCouple(i);
        if (!couple->IsUsed()) continue;
        std::fprintf(stderr, "  CUTS mat=%d I=%.3f eV  gamma=%.1f eV  e-=%.1f eV  e+=%.1f eV\n",
                     mat_id, couple->GetMaterial()->GetIonisation()->GetMeanExcitationEnergy() / eV,
                     (*table->GetEnergyCutsVector(idxG4GammaCut))[i] / eV,
                     (*table->GetEnergyCutsVector(idxG4ElectronCut))[i] / eV,
                     (*table->GetEnergyCutsVector(idxG4PositronCut))[i] / eV);
    }
}

// ---------------------------------------------------------------------------
// Detector construction: a single large box of the target material. The
// box is sized well beyond the positron's range at these energies -- the
// fixed STEP_CM path length is enforced in SteppingAction (by accumulated
// true path length), not by geometry, matching PENELOPE's treatment of an
// effectively infinite medium.
// ---------------------------------------------------------------------------
namespace { extern G4UserLimits *gLimits; }   // defined with the other per-event state below

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
    explicit DetectorConstruction(G4Material *mat) : fMaterial(mat) {}

    void SetMaterial(G4Material *mat) { fMaterial = mat; }

    G4VPhysicalVolume *Construct() override {
        const G4double half = 50.0 * cm;
        auto *solid = new G4Box("World", half, half, half);
        auto *logic = new G4LogicalVolume(solid, fMaterial, "World");
        if (gLimits) logic->SetUserLimits(gLimits);   // same object re-attached on every rebuild
        return new G4PVPlacement(nullptr, G4ThreeVector(), logic, "World",
                                  nullptr, false, 0);
    }

private:
    G4Material *fMaterial;
};

// ---------------------------------------------------------------------------
// Primary generator: e+ at the box entrance, moving in +z. Energy is
// updated between BeamOn() calls for each energy point in the grid.
// ---------------------------------------------------------------------------
static const double kGunZ_cm = -49.9;   // start of every history; end positions are relative to it

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
    PrimaryGeneratorAction() {
        fGun = new G4ParticleGun(1);
        fGun->SetParticleDefinition(G4Positron::Definition());
        fGun->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
        fGun->SetParticlePosition(G4ThreeVector(0, 0, kGunZ_cm * cm));
    }
    ~PrimaryGeneratorAction() override { delete fGun; }

    void SetKineticEnergy(double e_ev) { fGun->SetParticleEnergy(e_ev * eV); }

    void GeneratePrimaries(G4Event *event) override {
        fGun->GeneratePrimaryVertex(event);
    }

private:
    G4ParticleGun *fGun;
};

// ---------------------------------------------------------------------------
// Per-event tracking state (single-threaded batch tool -- module-level
// state mirrors the PENELOPE Fortran driver's use of COMMON blocks).
// ---------------------------------------------------------------------------
namespace {
struct HistoryResult {
    double theta_rad = 0.0;
    double de_total_ev = 0.0;
    double dE_kin_ev = 0.0;
    double e_final_ev = 0.0;
    int annihilated = 0;
    double z_cm = 0.0;            // end position along the initial direction, from the start
    double r_cm = 0.0;            // end distance from the initial axis
    double dphi_rad = 0.0;        // azimuth of the final direction about +z, from the lateral displacement
};

HistoryResult gResult;
double gStepCmTarget = 0.0;   // config: target path length [cm]
double gE0_eV = 0.0;          // current energy point [eV]
double gE0_keV = 0.0;         // current energy point [keV], for output
int gMatId = 0;               // current material id, for output
double gCumPathLen_mm = 0.0;  // primary track's accumulated true path length
G4ThreeVector gLastDir(0, 0, 1);
std::ofstream *gOut = nullptr;
long gSurvCount = 0;          // survivors (annihilated == 0) at this energy
double kStopEnergyEV = 1.0;   // "reached rest" threshold; set from LOWEST_E_EV in main()
G4UserLimits *gLimits = nullptr;   // max allowed (true) step, updated every step
int gStepLimit = 1;
double gMaxStepCap_mm = 0.0;  // STEP_CM * MAX_STEP_FRAC, in mm
const double kPathEps_mm = 1.0e-6;  // 1 nm: "remaining path is zero"

double clamp_cosine(double w) {
    if (w > 1.0) return 1.0;
    if (w < -1.0) return -1.0;
    return w;
}
}  // namespace

class SteppingAction : public G4UserSteppingAction {
public:
    void UserSteppingAction(const G4Step *step) override {
        auto *track = step->GetTrack();

        // Only the primary positron is ever stepped here -- secondaries
        // (delta electrons, bremsstrahlung/annihilation photons) are killed
        // at production by StackingAction, so this step's deposit is purely
        // the primary's own continuous energy loss below the production
        // threshold. Informational total-deposit column, not read by
        // build_tables.py.
        gResult.de_total_ev += step->GetTotalEnergyDeposit() / eV;

        gCumPathLen_mm += step->GetStepLength() / mm;
        double ke_ev = track->GetKineticEnergy() / eV;
        if (ke_ev > kStopEnergyEV) {
            gLastDir = track->GetMomentumDirection();
        }

        bool stopped = (ke_ev <= kStopEnergyEV) ||
                       track->GetTrackStatus() == fStopAndKill ||
                       track->GetTrackStatus() == fStopButAlive;
        // Remaining true path to the fixed step length. With STEP_LIMIT=1 the
        // G4StepLimiter is told, after every step, that the next step may not
        // exceed this remainder, so the final step ends exactly at STEP_CM
        // (the analogue of PENELOPE's JUMP(DSMAX) with DSMAX = STEP_CM-DSTOT).
        double remaining_mm = gStepCmTarget * 10.0 - gCumPathLen_mm;   // cm->mm
        bool reached_step = (remaining_mm <= kPathEps_mm);
        if (!stopped && !reached_step && gStepLimit) {
            double cap = remaining_mm;
            if (gMaxStepCap_mm > 0.0 && gMaxStepCap_mm < cap) cap = gMaxStepCap_mm;
            gLimits->SetMaxAllowedStep(cap * mm);
        }

        if (stopped || reached_step) {
            const G4ThreeVector pos = step->GetPostStepPoint()->GetPosition();
            gResult.z_cm = pos.z() / cm - kGunZ_cm;
            gResult.r_cm = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y()) / cm;
            // dphi of the direction used for theta (gLastDir), as the PENELOPE driver:
            // atan2(x v - y u, x u + y v) in (-pi, pi], 0 when either transverse vector vanishes
            const double x = pos.x(), y = pos.y(), u = gLastDir.x(), v = gLastDir.y();
            gResult.dphi_rad = (x * x + y * y > 0.0 && u * u + v * v > 0.0)
                                   ? std::atan2(x * v - y * u, x * u + y * v) : 0.0;
        }
        if (stopped) {
            gResult.annihilated = 1;
            gResult.e_final_ev = 0.0;
            gResult.dE_kin_ev = gE0_eV;
            gResult.theta_rad = std::acos(clamp_cosine(gLastDir.z()));
            track->SetTrackStatus(fStopAndKill);
        } else if (reached_step) {
            gResult.annihilated = 0;
            gResult.e_final_ev = ke_ev;
            gResult.dE_kin_ev = gE0_eV - ke_ev;
            gResult.theta_rad = std::acos(clamp_cosine(gLastDir.z()));
            track->SetTrackStatus(fStopAndKill);
        }
    }
};

// Kill every secondary at production instead of transporting it. Matches
// PENELOPE's driver, which tracks only the primary positron per history and
// treats secondaries (delta electrons, bremsstrahlung/annihilation photons)
// as absorbed on the spot -- so the simulation only ever moves a positron,
// never an electron or a photon.
class StackingAction : public G4UserStackingAction {
public:
    G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track *track) override {
        if (track->GetParentID() == 0) return fUrgent;
        gResult.de_total_ev += track->GetKineticEnergy() / eV;
        return fKill;
    }
};

class EventAction : public G4UserEventAction {
public:
    void BeginOfEventAction(const G4Event *) override {
        gResult = HistoryResult{};
        gCumPathLen_mm = 0.0;
        gLastDir = G4ThreeVector(0, 0, 1);
        if (gStepLimit && gLimits) {
            double cap_mm = gStepCmTarget * 10.0;
            if (gMaxStepCap_mm > 0.0 && gMaxStepCap_mm < cap_mm) cap_mm = gMaxStepCap_mm;
            gLimits->SetMaxAllowedStep(cap_mm * mm);
        }
    }

    void EndOfEventAction(const G4Event *) override {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "%11.4f %10.6f %14.6e %14.6e %14.6e %1d %2d %11.7f %11.7f %10.6f\n",
                      gE0_keV, gResult.theta_rad, gResult.de_total_ev,
                      gResult.dE_kin_ev, gResult.e_final_ev,
                      gResult.annihilated, gMatId, gResult.z_cm, gResult.r_cm,
                      gResult.dphi_rad);
        (*gOut) << buf;
        if (gResult.annihilated == 0) gSurvCount++;
    }
};

class ActionInitialization : public G4VUserActionInitialization {
public:
    explicit ActionInitialization(PrimaryGeneratorAction *gen) : fGen(gen) {}

    void Build() const override {
        SetUserAction(fGen);
        SetUserAction(new SteppingAction());
        SetUserAction(new EventAction());
        SetUserAction(new StackingAction());
    }

private:
    PrimaryGeneratorAction *fGen;
};

// ---------------------------------------------------------------------------
// Main: discover materials, loop material x energy, write raw text.
// ---------------------------------------------------------------------------
int main() {
    std::string cfg_path = "input/sim.in";
    if (const char *env = std::getenv("TABLEGEN_IN")) {
        if (*env) cfg_path = env;
    }
    Config cfg = read_config(cfg_path);
    gStepCmTarget = cfg.step_cm;

    std::vector<std::string> mat_files;
    for (int i = 1; i <= 200; i++) {
        std::string p = cfg.mat_dir + "/mat" + std::to_string(i) + ".mat";
        std::ifstream test(p);
        if (!test.good()) break;
        mat_files.push_back(p);
    }
    if (mat_files.empty()) {
        std::cerr << "ERROR: no .mat files found in " << cfg.mat_dir << "\n";
        return 1;
    }
    for (int m : cfg.mat_list) {
        if (m > static_cast<int>(mat_files.size())) {
            std::cerr << "ERROR: MAT_LIST entry " << m << " > NMATS = "
                      << mat_files.size() << "\n";
            return 1;
        }
    }
    const std::string raw_dir = cfg.subset() ? "output/geant4/raw_extra"
                                             : "output/geant4/raw";
    if (cfg.subset()) {
        if (::mkdir(raw_dir.c_str(), 0775) != 0 && errno != EEXIST) {
            std::cerr << "ERROR: cannot create " << raw_dir << "\n";
            return 1;
        }
    }

    std::vector<double> energies(cfg.n_energies);
    double fac = std::log(cfg.emax_ev / cfg.emin_ev) / (cfg.n_energies - 1);
    for (int i = 0; i < cfg.n_energies; i++) {
        energies[i] = cfg.emin_ev * std::exp(i * fac);
    }

    std::cerr << "run_geant4 configuration:\n"
              << "  CONFIG    = " << cfg_path << "\n"
              << "  STEP_CM   = " << cfg.step_cm << " cm\n"
              << "  EMIN_SIM  = " << cfg.emin_ev << " eV\n"
              << "  EMAX_SIM  = " << cfg.emax_ev << " eV\n"
              << "  NENRG     = " << cfg.n_energies << "\n"
              << "  NHIST     = " << cfg.n_hist << "\n"
              << "  MAT_DIR   = " << cfg.mat_dir << "\n"
              << "  NMATS     = " << mat_files.size() << "\n"
              << "  NSURV_MIN = " << cfg.nsurv_min << "\n"
              << "  NHIST_MAX = " << cfg.nhist_max << "\n"
              << "  SEED      = " << cfg.seed << "\n"
              << "  PHYSICS   = " << cfg.physics << "  STEP_LIMIT=" << cfg.step_limit
              << "  LOWEST_E_EV=" << cfg.lowest_e_ev << "  CUT_FLOOR_EV=" << cfg.cut_floor_ev
              << "  CUT_UM=" << cfg.cut_um << "  MAX_STEP_FRAC=" << cfg.max_step_frac << "\n";
    if (cfg.subset()) {
        std::cerr << "  SUBSET RUN: output to " << raw_dir << "/matN_"
                  << cfg.out_tag << ".dat\n  MAT_LIST     =";
        for (int m : cfg.mat_list) std::cerr << " " << m;
        std::cerr << "\n  E_INDEX_LIST (0-based) =";
        for (int i : cfg.e_index_list) std::cerr << " " << i;
        std::cerr << "\n";
    }
    std::cerr << "--------------------------------------------\n";

    // A single G4RunManager is created for the whole job and reused across
    // materials -- deleting and recreating G4RunManager per material
    // segfaults on the second iteration (physics processes get re-registered
    // on the globally-shared G4ParticleTable). Between materials we swap the
    // detector's material and tell the kernel geometry/physics changed, so
    // it rebuilds solids/volumes and dE/dx tables for the new material.
    MatSpec spec0 = parse_penelope_mat(mat_files[0]);
    G4Material *material0 = build_g4_material(spec0, 1);
    double cur_rho = spec0.density_g_cm3;        // of the material being simulated, for the config line

    auto *runManager = new G4RunManager();
    auto *detector = new DetectorConstruction(material0);
    runManager->SetUserInitialization(detector);

    // EM physics: G4EmLivermorePhysics (default, PHYSICS LIV), or
    // G4EmStandardPhysics_option4 (PHYSICS OPT4; Goudsmit-Saunderson msc with
    // Mott correction for e+/e-, Seltzer-Berger bremsstrahlung, tables from
    // 100 eV, G4PenelopeIonisationModel for e+ below 100 keV) or option3
    // (PHYSICS OPT3; Urban msc, Moller-Bhabha ionisation throughout).
    // Livermore was chosen after a 30-material comparison against the
    // PENELOPE driver (Oct 2026): the 50%-annihilation energy agrees within
    // 1% for every material (option4: up to 5%), the survivors' mean dE_kin
    // within ~1% on average (option4 ~2.4%) and mean theta within ~3%
    // (option4 similar; option3 ~12% on material 13).
    // G4StepLimiterPhysics adds the G4StepLimiter process that enforces the
    // per-step maximum set in SteppingAction/EventAction.
    auto *physicsList = new G4VModularPhysicsList();
    if (cfg.physics == "OPT3")      physicsList->RegisterPhysics(new G4EmStandardPhysics_option3());
    else if (cfg.physics == "OPT4") physicsList->RegisterPhysics(new G4EmStandardPhysics_option4());
    else                            physicsList->RegisterPhysics(new G4EmLivermorePhysics());
    if (cfg.step_limit) physicsList->RegisterPhysics(new G4StepLimiterPhysics());
    // Range cut for secondary production (they are killed anyway, but the
    // threshold sets the hard/soft split of the energy-loss straggling, as
    // WCC/WCR = 100 eV do in the PENELOPE driver). The threshold table's lower
    // edge must be lowered from its 990 eV default for the cut to reach ~100 eV.
    physicsList->SetDefaultCutValue(cfg.cut_um * um);
    G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(cfg.cut_floor_ev * eV, 10.0 * GeV);
    runManager->SetUserInitialization(physicsList);

    // Tracking cut: PENELOPE follows the positron down to EABS = 50 eV; the
    // standard lists stop e+/e- at 1 keV by default and annihilate at rest.
    auto *em = G4EmParameters::Instance();
    em->SetLowestElectronEnergy(cfg.lowest_e_ev * eV);
    em->SetMinEnergy(100.0 * eV);
    kStopEnergyEV = cfg.lowest_e_ev;
    gStepLimit = cfg.step_limit;
    gMaxStepCap_mm = (cfg.max_step_frac < 1.0) ? cfg.step_cm * 10.0 * cfg.max_step_frac : 0.0;
    gLimits = new G4UserLimits(cfg.step_cm * cm);

    auto *primaryGen = new PrimaryGeneratorAction();
    runManager->SetUserInitialization(new ActionInitialization(primaryGen));

    runManager->Initialize();

    for (size_t im = 0; im < mat_files.size(); im++) {
        int mat_id = static_cast<int>(im) + 1;
        if (!cfg.mat_list.empty() && !cfg.mat_list.count(mat_id)) continue;
        std::cerr << " Processing material " << mat_id << "\n";

        // material0 was built from mat_files[0]; any other material needs
        // the swap, also when earlier materials were skipped by MAT_LIST.
        if (im > 0) {
            MatSpec spec = parse_penelope_mat(mat_files[im]);
            G4Material *material = build_g4_material(spec, mat_id);
            cur_rho = spec.density_g_cm3;
            detector->SetMaterial(material);
            runManager->ReinitializeGeometry(true);
            runManager->PhysicsHasBeenModified();
        }

        // One CLHEP stream per material, seeded with SEED + mat_id (the
        // PENELOPE driver seeds RANECU the same way), consumed continuously
        // across that material's energies and extra batches, so each
        // material's output is a deterministic function of (config, SEED,
        // mat_id), independent of MAT_LIST.
        CLHEP::HepRandom::setTheSeed(cfg.seed + mat_id);

        std::string raw_path = raw_dir + "/mat" + std::to_string(mat_id) +
                               (cfg.subset() ? "_" + cfg.out_tag : "") + ".dat";
        std::ofstream out(raw_path);
        if (!out) {
            std::cerr << "ERROR: cannot open " << raw_path << " for writing\n";
            return 1;
        }
        // First line: the run configuration (build_tables.py reads it).
        out << std::setprecision(10)
            << "# config: engine=geant4 step_cm=" << cfg.step_cm
            << " emin_eV=" << cfg.emin_ev << " emax_eV=" << cfg.emax_ev
            << " nenrg=" << cfg.n_energies << " nhist=" << cfg.n_hist
            << " seed=" << cfg.seed << " physics=" << cfg.physics
            << " engine_version=" << G4VERSION_NUMBER / 100 << "."
            << (G4VERSION_NUMBER / 10) % 10 << "." << G4VERSION_NUMBER % 10
            << " rho_gcm3=" << cur_rho << "\n";
        out << "# E0_keV theta_rad DE_total_eV dE_kin_eV E_final_eV"
            << " annihil_flag matID z_cm r_cm dphi_rad\n";
        gOut = &out;
        gMatId = mat_id;
        const int first_ie = cfg.e_index_list.empty() ? 0 : *cfg.e_index_list.begin();

        for (int ie = 0; ie < cfg.n_energies; ie++) {
            if (!cfg.e_index_list.empty() && !cfg.e_index_list.count(ie)) continue;
            gE0_eV = energies[ie];
            gE0_keV = gE0_eV / 1000.0;
            primaryGen->SetKineticEnergy(gE0_eV);

            // Adaptive batches: first batch of NHIST always; further
            // batches while survivors < NSURV_MIN and histories <
            // NHIST_MAX; the last batch is clipped to the cap. With the
            // defaults this is exactly one BeamOn(NHIST) as before.
            long n_done = 0;
            gSurvCount = 0;
            do {
                long batch = cfg.n_hist;
                if (n_done + batch > cfg.nhist_max) batch = cfg.nhist_max - n_done;
                if (batch <= 0) break;
                runManager->BeamOn(static_cast<int>(batch));
                if (n_done == 0 && ie == first_ie) print_cuts(mat_id);
                n_done += batch;
            } while (cfg.nsurv_min > 0 && gSurvCount < cfg.nsurv_min &&
                     n_done < cfg.nhist_max);
            if (cfg.nsurv_min > 0 || cfg.subset()) {
                std::fprintf(stderr, "   i0=%4d E0=%12.5E eV  n_hist=%10ld  n_surv=%10ld\n",
                             ie, gE0_eV, n_done, gSurvCount);
            }
        }

        out.close();
        gOut = nullptr;
        std::cerr << "  Done material " << mat_id << "\n";
    }

    delete runManager;
    std::cerr << "All done.\n";
    return 0;
}
