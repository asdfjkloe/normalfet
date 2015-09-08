#define ARMA_NO_DEBUG // no bound checks
//#define GNUPLOT_NOPLOTS

#include <armadillo>
#include <iostream>
#include <omp.h>
#include <xmmintrin.h>

#define CHARGE_DENSITY_HPP_BODY

#include "charge_density.hpp"
#include "circuit.hpp"
#include "constant.hpp"
#include "contact.hpp"
#include "current.hpp"
#include "device.hpp"
#include "device_params.hpp"
#include "geometry.hpp"
#include "green.hpp"
#include "inverter.hpp"
#include "model.hpp"
#include "potential.hpp"
#include "ring_oscillator.hpp"
#include "voltage.hpp"
#include "wave_packet.hpp"
#include "util/movie.hpp"

#undef CHARGE_DENSITY_HPP_BODY

#include "charge_density.hpp"

using namespace arma;
using namespace std;

static inline void setup() {
    // disable nested parallelism globally
    omp_set_nested(0);

    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
}

// select device type
static inline device_params get_device_params(const string & n) {
    if (n == "nfet") {
        return nfet;
    }
    if (n == "nfetc") {
        return nfetc;
    }
    if (n == "pfet") {
        return pfet;
    }
    if (n == "pfetc") {
        return pfetc;
    }

    ifstream file(n);
    if (!file) {
        cout << "no device_params with name or filename " << n << endl;
        exit(0);
    }

    stringstream ss;
    ss << file.rdbuf();

    try {
        device_params p(ss.str());
        return p;
    } catch(...) {
        cout << "device_params file corrupted!" << endl;
        exit(0);
    }
}

static inline void output_results() {
    cout << endl << endl << "results" << endl << scientific << setprecision(15);
}

// creates dev files
static inline void dev() {
    ofstream file = ofstream("nfet.ini");
    file << nfet.to_string();
    file.close();
    file = ofstream("pfet.ini");
    file << pfet.to_string();
    file.close();
    file = ofstream("nfetc.ini");
    file << nfetc.to_string();
    file.close();
    file = ofstream("pfetc.ini");
    file << pfetc.to_string();
    file.close();
}

// calculate capacitance of device
static inline void cap(const device_params & p) {
    double csg = c::eps_0 * M_PI * (p.R * p. R - (p.r_cnt + p.d_ox) * (p.r_cnt + p.d_ox)) / p.l_sg * 1e-9;
    double cdg = c::eps_0 * M_PI * (p.R * p. R - (p.r_cnt + p.d_ox) * (p.r_cnt + p.d_ox)) / p.l_dg * 1e-9;
    double czyl = 2 * M_PI * c::eps_0 * p.eps_ox * p.l_g / std::log((p.r_cnt + p.d_ox) / p.r_cnt) * 1e-9;

    output_results();
    cout << "C_sg  = " << csg << endl;
    cout << "C_dg  = " << cdg << endl;
    cout << "C_zyl = " << czyl << endl;
    cout << "C_g   = " << csg + cdg + czyl << endl;
}

// computes 1D potential
static inline void pot1D(const device_params & p, const voltage<3> & V, bool nosc) {
    potential phi;
    if (nosc) {
        arma::vec R0 = potential::get_R0(p, V);
        phi = potential(p, R0);
    } else {
        device d("pot1Ddev", p, V);
        if (!d.steady_state()) {
            std::cout << "ERROR: steady_state not converged!!" << std::endl;
            return;
        }
        phi = d.phi[0];
    }

    output_results();
    for (unsigned i = 0; i < phi.data.size(); ++i) {
        cout << p.x[i] << " " << phi.data[i] << endl;
    }
}

// computes 2D potential
static inline void pot2D(const device_params & p, const voltage<3> & V, bool nosc) {
    arma::vec phi2Dv;

    if (nosc) {
        phi2Dv = spsolve(potential::get_C(p), potential::get_R0(p, V));
    } else {
        device d("pot2Ddev", p, V);
        if (!d.steady_state()) {
            std::cout << "ERROR: steady_state not converged!!" << std::endl;
            return;
        }
        phi2Dv = spsolve(potential::get_C(p), potential::get_R(p, potential::get_R0(p, V), d.n[0]));
    }

    arma::mat phi2D(p.N_x, p.M_r);

    // fill the matrix
    for (int j = 0; j < p.M_r; ++j) {
        for (int i = 0; i < p.N_x; ++i) {
            phi2D(i, j) = phi2Dv(j * p.N_x + i);
        }
    }
    // mirror
    phi2D = arma::join_horiz(arma::fliplr(phi2D), phi2D);
    arma::vec r = arma::join_vert(arma::flipud(-p.r), p.r);

    output_results();
    for (int i = 0; i < p.N_x; ++i) {
        for (int j = 0; j < 2 * p.M_r; ++j) {
            cout << p.x(i) << " " << r(j) << " " << phi2D(i, j) << endl;
        }
        cout << endl;
    }
}

// computes local density of states
static inline void ldos(const device_params & p, const voltage<3> & V, double E0, double E1, int N, bool nosc) {
    arma::vec E = arma::linspace(E0, E1, N);

    potential phi;
    if (nosc) {
        arma::vec R0 = potential::get_R0(p, V);
        phi = potential(p, R0);
    } else {
        device d("ldosdev", p, V);
        if (!d.steady_state()) {
            std::cout << "ERROR: steady_state not converged!!" << std::endl;
            return;
        }
        phi = d.phi[0];
    }

    arma::mat lDOS = get_lDOS(p, phi, N, E);

    output_results();
    for (int i = 0; i < p.N_x; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << p.x(i) << " " << E(j) << " " << lDOS(j, i) << endl;
        }
        cout << endl;
    }
}

// computes charge density
static inline void charge(const device_params & p, const voltage<3> & V) {
    device d("pot2Ddev", p, V);
    if (!d.steady_state()) {
        std::cout << "ERROR: steady_state not converged!!" << std::endl;
        return;
    }

    output_results();
    for (unsigned i = 0; i < d.n[0].total.size(); ++i) {
        cout << p.x[i] << " " << d.n[0].total[i] / p.dx << endl;
    }
}

// computes the current for a given voltage point
static inline void point(const device_params & p, const voltage<3> & V) {
    device d("pointdev", p, V);
    if (!d.steady_state()) {
        std::cout << "ERROR: steady_state not converged!!" << std::endl;
        return;
    }

    output_results();
    cout << d.I[0].total[0] << std::endl;
}

// compute transfer curve
static inline void trans(const device_params & p, const voltage<3> & V0, double V_g1, int N) {
    auto curve = transfer(p, {V0}, V_g1, N);

    output_results();
    for (unsigned i = 0; i < curve.n_rows; ++i) {
        if (curve(i, 1) != 666) {
            cout << curve(i, 0) << " " << curve(i, 1) << endl;
        }
    }
}

// compute output curve
static inline void outp(const device_params & p, const voltage<3> & V0, double V_d1, int N) {
    auto curve = output(p, {V0}, V_d1, N);

    output_results();
    for (unsigned i = 0; i < curve.n_rows; ++i) {
        if (curve(i, 1) != 666) {
            cout << curve(i, 0) << " " << curve(i, 1) << endl;
        }
    }
}

// compute steady state inverter
static inline void inv(const device_params & p1, const device_params & p2, const voltage<3> & V0, double V_in1, int N) {
    inverter inv(p1, p2);
    vec V_in0 = linspace(V0[2], V_in1, N);
    vec V_in(N);
    vec V_out(N);

    int j = 0;
    for (int i = 0; i < N; ++i) {
        cout << "\nstep " << i+1 << "/" << N << ": \n";
        if (inv.steady_state({ V0[0], V0[1], V_in0(i) })) {
            V_in(j) = V_in0(i);
            V_out(j++) = inv.get_output(0)->V;
        }
    }
    V_in.resize(j);
    V_out.resize(j);

    output_results();
    for (int i = 0; i < j; ++i) {
        cout << V_in(i) << " " << V_out(i) << endl;
    }
}

// compute initial wave function
static inline void wave(const device_params & p, const voltage<3> V, double E, bool src) {
    device d("wavedev", p, V);

    if (!d.steady_state()) {
        std::cout << "ERROR: steady_state not converged!!" << std::endl;
        return;
    }

    cx_double Sigma_s, Sigma_d;
    cx_vec G;
    if (src) {
        G = green_col<true>(p, d.phi[0], E, Sigma_s, Sigma_d);
    } else {
        G = green_col<false>(p, d.phi[0], E, Sigma_s, Sigma_d);
    }

    // norm
    G /= arma::max(arma::abs(G));

    output_results();
    for (unsigned i = 0; i < G.size() / 2; ++i) {
        cout << p.x(i) << " " << std::real(G(2 * i)) << " " << std::imag(G(2 * i)) << endl;
    }
}

// compute time_evolution for step like signal
static inline void step(const device_params & p, const voltage<3> V0, const voltage<3> V1, double tswitch, double T) {
    signal<3> s = linear_signal<3>(tswitch, V0, V1) + signal<3>(T - tswitch, V1);

    device d = device("stepdev", p, V0);
    if (!d.steady_state()) {
        std::cout << "ERROR: steady_state not converged!!" << std::endl;
        return;
    }

    d.init_time_evolution(s.N_t);

    auto Is = vec(s.N_t);
    auto Id = vec(s.N_t);
    Is(0) = d.I[0].total(0);
    Id(0) = d.I[0].total(d.I[0].total.size());

    for (int i = 1; i < s.N_t; ++i) {
        d.contacts[S]->V = s[i][S];
        d.contacts[D]->V = s[i][D];
        d.contacts[G]->V = s[i][G];
        if (!d.time_step()) {
            std::cout << "ERROR: time_step not converged!!" << std::endl;
            return;
        }
        Is(i) = d.I[i].total(0);
        Id(i) = d.I[i].total(d.I[i].total.size());
    }

    output_results();
    for (int i = 0; i < s.N_t; ++i) {
        cout << i * c::dt << " " << s[i][S] << " " <<  s[i][D] << " " << s[i][G] << " " << Is(i) << " " << Id(i) <<  endl;
    }
}

// ring oscillator
inline void ro(const device_params & p1, const device_params & p2, double V_ss, double V_dd, double T, double C) {
    ring_oscillator<3> ro(p1, p2, C);
    ro.time_evolution(signal<2>(T, voltage<2>{ V_ss, V_dd }));

    output_results();
    for (unsigned i = 0; i < ro.V_out.size(); ++i) {
        cout << i * c::dt << " " << ro.V_out[i][0] << " " << ro.V_out[i][1] << " " << ro.V_out[i][2] << endl;
    }
}

// inverter square signal
inline void inv_square(const device_params & p1, const device_params & p2, const voltage<3> & V, double V_g1, double tswitch, double f, int N, double C) {
    auto s = square_signal<3>(N / f, V, {V[S], V[D], V_g1}, f, tswitch, tswitch);

    inverter inv(p1, p2, C);
    inv.time_evolution(s);

    output_results();
    for (int i = 0; i < s.N_t; ++i) {
        cout << i * c::dt << " " << s[i][G] << " " << inv.V_out[i][0] << endl;
    }
}

inline void test() {
    device d1 = device("nfet", nfet, { 0.0, 0.2, -0.2});
    d1.steady_state();
    plot_ldos(nfet, d1.phi[0], 1000, -1.5, 1.5);

    device d2 = device("nfetc", nfetc, { 0.0, 0.2, -0.2});
    d2.steady_state();
    plot_ldos(nfetc, d2.phi[0], 1000, -1.5, 1.5);

    auto curve0 = transfer(nfet, {{0.0, 0.2, -0.2}}, 0.2, 100);
    auto curve1 = arma::mat(curve0.n_rows, curve0.n_cols);
    auto curve1a = arma::mat(curve0.n_rows, curve0.n_cols);
    int j = 0;
    for (unsigned i = 0; i < curve0.n_rows; ++i) {
        if (curve0(i, 1) != 666) {
            curve1(j, 0) = curve0(i, 0);
            curve1(j, 1) = std::log(curve0(i, 1));
            curve1a(j,0) = curve0(i, 0);
            curve1a(j,1) = curve0(i, 1);
            ++j;
        }
    }
    curve1.resize(j, 2);
    curve1a.resize(j, 2);

    auto curve2 = transfer(nfetc, {{0.0, 0.2, -0.2}}, 0.2, 100);
    auto curve3 = arma::mat(curve2.n_rows, curve2.n_cols);
    auto curve3a = arma::mat(curve2.n_rows, curve2.n_cols);
    j = 0;
    for (unsigned i = 0; i < curve2.n_rows; ++i) {
        if (curve2(i, 1) != 666) {
            curve3(j, 0) = curve2(i, 0);
            curve3(j, 1) = std::log(curve2(i, 1));
            curve3a(j,0) = curve2(i, 0);
            curve3a(j,1) = curve2(i, 1);
            ++j;
        }
    }
    curve3.resize(j, 2);
    curve3a.resize(j, 2);

    plot(make_pair(curve1.col(0), curve1.col(1)), make_pair(curve3.col(0), curve3.col(1)));
    plot(make_pair(curve1a.col(0), curve1a.col(1)), make_pair(curve3a.col(0), curve3a.col(1)));
}

int main(int argc, char ** argv) {
    setup();

    // num threads
    if (argc > 1) {
        omp_set_num_threads(stoi(argv[1]));
    }

    string stype = "";
    if (argc > 2) {
        stype = argv[2];
    }

    // second argument chooses the type of simulation
    if (stype == "dev") {
        dev();
    } else if (stype == "cap") {
        device_params p = get_device_params(argv[3]);
        cap(p);
    } else if (stype == "pot1D") {
        if ((argc != 7) && (argc != 8)) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        bool nosc = ((argc == 8) && (string(argv[7]) == "nosc"));
        pot1D(p, {V_s, V_d, V_g}, nosc);
    } else if (stype == "pot2D") {
        if ((argc != 7) && (argc != 8)) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        bool nosc = ((argc == 8) && (string(argv[7]) == "nosc"));
        pot2D(p, {V_s, V_d, V_g}, nosc);
    } else if (stype == "ldos") {
        if ((argc != 10) && (argc != 11)) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        double E0 = stod(argv[7]);
        double E1 = stod(argv[8]);
        double N = stod(argv[9]);
        bool nosc = ((argc == 11) && (string(argv[10]) == "nosc"));
        ldos(p, {V_s, V_d, V_g}, E0, E1, N, nosc);
    } else if (stype == "charge") {
        if (argc != 7) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        charge(p, {V_s, V_d, V_g});
    } else if (stype == "point") {
        if (argc != 7) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        point(p, {V_s, V_d, V_g});
    } else if (stype == "trans") {
        if (argc != 9) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p = get_device_params(argv[3]);
        double V_s0 = stod(argv[4]);
        double V_d0 = stod(argv[5]);
        double V_g0 = stod(argv[6]);
        double V_g1 = stod(argv[7]);
        int N = stoi(argv[8]);
        trans(p, {V_s0, V_d0, V_g0}, V_g1, N);
    } else if (stype == "outp") {
        if (argc != 9) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p = get_device_params(argv[3]);
        double V_s0 = stod(argv[4]);
        double V_d0 = stod(argv[5]);
        double V_g0 = stod(argv[6]);
        double V_d1 = stod(argv[7]);
        int N = stoi(argv[8]);
        outp(p, {V_s0, V_d0, V_g0}, V_d1, N);
    } else if (stype == "inv") {
        if (argc != 10) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p1 = get_device_params(argv[3]);
        device_params p2 = get_device_params(argv[4]);
        double V_ss = stod(argv[5]);
        double V_dd = stod(argv[6]);
        double V_in0 = stod(argv[7]);
        double V_in1 = stod(argv[8]);
        int N = stoi(argv[9]);
        inv(p1, p2, {V_ss, V_dd, V_in0}, V_in1, N);
    } else if (stype == "wave") {
        if (argc != 9) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        double E = stod(argv[7]);
        bool src = (string(argv[8]) == "s");
        wave(p, {V_s, V_d, V_g}, E, src);
    } else if (stype == "step") {
        if (argc != 12) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p = get_device_params(argv[3]);
        double V_s0 = stod(argv[4]);
        double V_d0 = stod(argv[5]);
        double V_g0 = stod(argv[6]);
        double V_s1 = stod(argv[7]);
        double V_d1 = stod(argv[8]);
        double V_g1 = stod(argv[9]);
        double tswitch = stod(argv[10]);
        double T = stod(argv[11]);
        step(p, {V_s0, V_d0, V_g0}, {V_s1, V_d1, V_g1}, tswitch, T);
    } else if (stype == "ro") {
        if (argc != 9) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p1 = get_device_params(argv[3]);
        device_params p2 = get_device_params(argv[4]);
        double V_ss = stod(argv[5]);
        double V_dd = stod(argv[6]);
        double T = stod(argv[7]);
        double C = stod(argv[8]);
        ro(p1, p2, V_ss, V_dd, T, C);
    } else if (stype == "inv_square") {
        if (argc != 13) {
            cout << "Wrong number of arguments!" << endl;
            return 0;
        }
        device_params p1 = get_device_params(argv[3]);
        device_params p2 = get_device_params(argv[4]);
        double V_s0 = stod(argv[5]);
        double V_d0 = stod(argv[6]);
        double V_g0 = stod(argv[7]);
        double V_g1 = stod(argv[8]);
        double tswitch = stod(argv[9]);
        double f = stod(argv[10]);
        int N = stoi(argv[11]);
        double C = stod(argv[12]);
        inv_square(p1, p2, {V_s0, V_d0, V_g0}, V_g1, tswitch, f, N, C);
    } else {
        test();
    }
}
