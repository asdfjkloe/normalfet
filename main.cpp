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
    } else if (n == "nfetc") {
        return nfetc;
    } else if (n == "pfet") {
        return pfet;
    } else if (n == "pfetc") {
        return pfetc;
    }

    //LOAD DEV

    throw new invalid_argument("no device_params with name" + n);
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

    cout << endl << endl << "results" << endl;
    for (unsigned i = 0; i < phi.data.size(); ++i) {
        cout << setprecision(12) << p.x[i] << " " << setprecision(12) << phi.data[i] << endl;
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
    phi2D = arma::join_horiz(arma::fliplr(phi2D), phi2D).t();

    cout << endl << endl << "results" << endl;
    for (unsigned j = 0; j < phi2D.n_cols; ++j) {
        for (unsigned i = 0; i < phi2D.n_rows; ++i) {
            cout << setprecision(12) << p.x[i] << " " << setprecision(12) << p.r[j] << " " << setprecision(12) << phi2D(j, i) << endl;
        }
        cout << endl;
    }
}

// computes local density of states
static inline void ldos(const device_params & p, const voltage<3> & V, double E0, double E1, int N, bool nosc) {
    arma::vec E = arma::linspace(E0, E1, N);

    device d("ldosdev", p, V);
    if (nosc) {
        arma::vec R0 = potential::get_R0(p, V);
        d.phi[0] = potential(p, R0);
    } else {
        if (!d.steady_state()) {
            std::cout << "ERROR: steady_state not converged!!" << std::endl;
            return;
        }
    }

    arma::mat lDOS = get_lDOS(p, d.phi[0], N, E);

    cout << endl << endl << "results" << endl;
    for (unsigned j = 0; j < lDOS.n_cols; ++j) {
        for (unsigned i = 0; i < lDOS.n_rows; ++i) {
            cout << setprecision(12) << p.x[i] << " " << setprecision(12) << p.r[j] << " " << setprecision(12) << lDOS(j, i) << endl;
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

    cout << endl << endl << "results" << endl;
    for (unsigned i = 0; i < d.n[0].total.size(); ++i) {
        cout << setprecision(12) << p.x[i] << " " << setprecision(12) << d.n[0].total[i] << endl;
    }
}

// computes the current for a given voltage point
static inline void point(const device_params & p, const voltage<3> & V) {
    device d("pointdev", p, V);
    if (!d.steady_state()) {
        std::cout << "ERROR: steady_state not converged!!" << std::endl;
        return;
    }
    cout << endl << endl << "results" << endl;
    cout << setprecision(12) << d.I[0].total[0] << std::endl;
}

// compute transfer curve
static inline void trans(const device_params & p, const voltage<3> & V0, double V_g1, int N) {
    auto curve = transfer(p, {V0}, V_g1, N);

    cout << endl << endl << "results" << endl;
    for (unsigned i = 0; i < curve.n_rows; ++i) {
        if (curve(i, 1) != 666) {
            cout << setprecision(12) << curve(i, 0) << " " << setprecision(12) << curve(i, 1) << endl;
        }
    }
}

// compute output curve
static inline void outp(const device_params & p, const voltage<3> & V0, double V_d1, int N) {
    auto curve = output(p, {V0}, V_d1, N);

    cout << endl << endl << "results" << endl;
    for (unsigned i = 0; i < curve.n_rows; ++i) {
        if (curve(i, 1) != 666) {
            cout << setprecision(12) << curve(i, 0) << " " << setprecision(12) << curve(i, 1) << endl;
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

    cout << endl << endl << "results" << endl;
    for (int i = 0; i < j; ++i) {
        cout << setprecision(12) << V_in(i) << " " << setprecision(12) << V_out(i) << endl;
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

    cout << endl << endl << "results" << endl;
    for (unsigned i = 0; i < G.size(); ++i) {
        cout << setprecision(12) << p.x(i) << " " << setprecision(12) << std::real(G(i)) << " " << setprecision(12) << std::imag(G(i)) << endl;
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

    cout << endl << endl << "results" << endl;
    for (int i = 0; i < s.N_t; ++i) {
        cout << setprecision(12) << s[i][S] << " " << setprecision(12) <<  s[i][D] << " " << setprecision(12) << s[i][G] << " " << setprecision(12) << Is(i) << " " << setprecision(12) << Id(i) <<  endl;
    }
}

// ring oscillator
inline void ro(const device_params & p1, const device_params & p2, double V_ss, double V_dd, double T, double C) {
    ring_oscillator<3> ro(p1, p2, C);
    ro.time_evolution(signal<2>(T, voltage<2>{ V_ss, V_dd }));

    cout << endl << endl << "results" << endl;
    for (unsigned i = 0; i < ro.V_out.size(); ++i) {
        cout << setprecision(12) << ro.V_out[i][0] << " " << setprecision(12) << ro.V_out[i][1] << " " << setprecision(12) << ro.V_out[i][2] << endl;
    }
}

// inverter square signal
inline void inv_square(const device_params & p1, const device_params & p2, const voltage<3> & V, double V_g1, double tswitch, double f, int N, double C) {
    auto s = square_signal<3>(N / f, V, {V[S], V[D], V_g1}, f, tswitch, tswitch);

    inverter inv(p1, p2, C);
    inv.time_evolution(s);

    cout << endl << endl << "results" << endl;
    for (int i = 0; i < s.N_t; ++i) {
        cout << setprecision(12) << s[i][G] << " " << setprecision(12) << inv.V_out[i][0] << endl;
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
    if (stype == "pot1D" && (argc == 7 || argc == 8)) {
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        bool nosc = ((argc == 8) && (string(argv[7]) == "nosc"));
        pot1D(p, {V_s, V_d, V_g}, nosc);
    } else if (stype == "pot2D" && (argc == 7 || argc == 8)) {
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        bool nosc = ((argc == 8) && (string(argv[7]) == "nosc"));
        pot2D(p, {V_s, V_d, V_g}, nosc);
    } else if (stype == "ldos" && (argc == 10 || argc == 11)) {
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        double E0 = stod(argv[7]);
        double E1 = stod(argv[8]);
        double N = stod(argv[9]);
        bool nosc = ((argc == 11) && (string(argv[10]) == "nosc"));
        ldos(p, {V_s, V_d, V_g}, E0, E1, N, nosc);
    } else if (stype == "charge" && argc == 7) {
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        charge(p, {V_s, V_d, V_g});
    } else if (stype == "point" && argc == 7) {
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        point(p, {V_s, V_d, V_g});
    } else if (stype == "trans" && argc == 9) {
        device_params p = get_device_params(argv[3]);
        double V_s0 = stod(argv[4]);
        double V_d0 = stod(argv[5]);
        double V_g0 = stod(argv[6]);
        double V_g1 = stod(argv[7]);
        int N = stoi(argv[8]);
        trans(p, {V_s0, V_d0, V_g0}, V_g1, N);
    } else if (stype == "outp" && argc == 9) {
        device_params p = get_device_params(argv[3]);
        double V_s0 = stod(argv[4]);
        double V_d0 = stod(argv[5]);
        double V_g0 = stod(argv[6]);
        double V_d1 = stod(argv[7]);
        int N = stoi(argv[8]);
        outp(p, {V_s0, V_d0, V_g0}, V_d1, N);
    } else if (stype == "inv" && argc == 10) {
        device_params p1 = get_device_params(argv[3]);
        device_params p2 = get_device_params(argv[4]);
        double V_ss = stod(argv[5]);
        double V_dd = stod(argv[6]);
        double V_in0 = stod(argv[7]);
        double V_in1 = stod(argv[8]);
        int N = stoi(argv[9]);
        inv(p1, p2, {V_ss, V_dd, V_in0}, V_in1, N);
    } else if (stype == "wave" && argc == 9) {
        device_params p = get_device_params(argv[3]);
        double V_s = stod(argv[4]);
        double V_d = stod(argv[5]);
        double V_g = stod(argv[6]);
        double E = stod(argv[7]);
        bool src = (string(argv[8]) == "s");
        wave(p, {V_s, V_d, V_g}, E, src);
    } else if (stype == "step" && argc == 12) {
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
    } else if (stype == "ro" && argc == 9) {
        device_params p1 = get_device_params(argv[3]);
        device_params p2 = get_device_params(argv[4]);
        double V_ss = stod(argv[5]);
        double V_dd = stod(argv[6]);
        double T = stod(argv[7]);
        double C = stod(argv[8]);
        ro(p1, p2, V_ss, V_dd, T, C);
    } else if (stype == "inv_square" && argc == 13) {
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

/*
// set the type of device (mosfet or tfet)
static device_params ntype = nfet;
static device_params ptype = pfet;


static inline void point(char ** argv) {
    // computes the current for a given voltage point

    double vs = stod(argv[3]);
    double vd = stod(argv[4]);
    double vg = stod(argv[5]);

    device d("ntype", ntype, {vs, vd, vg});
    d.steady_state();
    cout << "I = " << d.I[0].total[0] << std::endl;
}

static inline void trans(char ** argv) {
    // starts transfer-curve simulations with a certain gate-length

    double l_g = stod(argv[3]);
    double sox = stod(argv[4]);
    double Vg0 = stod(argv[5]);
    double Vg1 = stod(argv[6]);
    double Vd  = stod(argv[7]);
    int N      = stoi(argv[8]);

    stringstream ss;
    ss << "transfer/lg=" << l_g << "_lsox=" << sox << "_Vd=" << Vd;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    device d("prototype", ntype);
    d.p.l_g = l_g;

    // extract total contact length from prototype
    double tmp = d.p.l_sc + d.p.l_sox;
    d.p.l_sox = sox;
    d.p.l_sc = tmp - sox;
    d.p.update("updated");

    transfer<true>(d.p, { { 0, Vd, Vg0 } }, Vg1, N);

    ofstream s(save_folder() + "/device.ini");
    s << d.p.to_string();
    s.close();
}

static inline void outp(char ** argv) {
    // starts output-curve simulations with a certain gate-length

    double l_g = stod(argv[3]);
    double Vd0 = stod(argv[4]);
    double Vd1 = stod(argv[5]);
    double Vg  = stod(argv[6]);
    int N      = stoi(argv[7]);

    stringstream ss;
    ss << "output/lg=" << l_g << "_Vg=" << Vg;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    device d("prototype", ntype);
    d.p.l_g = l_g;
    d.p.update("updated");

    output<true>(d.p, { { 0, Vd0, Vg } }, Vd1, N);

    ofstream s(save_folder() + "/device.ini");
    s << d.p.to_string();
    s.close();
}

static void inv(char ** argv) {
    // starts a static inverter simulation

    double Vin0  = stod(argv[3]);
    double Vin1  = stod(argv[4]);
    double V_dd  = stod(argv[5]);
    int    N     = stoi(argv[6]);

    inverter inv(ntype, ptype);

    stringstream ss;
    ss << "ntd_inverter/Vdd=" << V_dd;

    cout << "saving results in " << save_folder(ss.str()) << endl;

    vec V_in = linspace(Vin0, Vin1, N);
    vec V_out(N);

    for (int i = 0; i < N; ++i) {
        cout << "\nstep " << i+1 << "/" << N << ": \n";
        inv.steady_state({ 0, V_dd, V_in(i) });
        V_out(i) = inv.get_output(0)->V;
    }

    mat ret = join_horiz(V_in, V_out);
    ret.save(save_folder() + "/inverter_curve.csv", csv_ascii);

    ofstream sn(save_folder() + "/ntype.ini");
    sn << ntype.to_string();
    sn.close();

    ofstream sp(save_folder() + "/ptype.ini");
    sp << ptype.to_string();
    sp.close();
}

static inline void ro(char ** argv) {
    // starts a transient ring-oscillator simulation

    double T = stod(argv[3]);
    double C = stod(argv[4]);
    double V_dd = stod(argv[5]);
    stringstream ss;
    ss << "ring_oscillator/" << "C=" << C << "_Vdd=" << V_dd;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    ring_oscillator<3> ro(ntype, ptype, C);
    ro.time_evolution(signal<2>(T, voltage<2>{ 0.0, V_dd }));
    ro.save<true>();

    ofstream sn(save_folder() + "/ntype.ini");
    sn << ntype.to_string();
    sn.close();

    ofstream sp(save_folder() + "/ptype.ini");
    sp << ptype.to_string();
    sp.close();
}

static inline void ldos(char ** argv) {
    // plots the ldos for a self-consistent static situation

    double vd = stod(argv[3]);
    double vg = stod(argv[4]);
    double Emin = stod(argv[5]);
    double Emax = stod(argv[6]);
    int    N    = stoi(argv[7]);

    device dev("test", ntype, voltage<3>{ 0, vd, vg });
    dev.steady_state();

    plot_ldos(dev.p, dev.phi[0], N, Emin, Emax);
}

static inline void pot(char ** argv) {
    // plots a self-consisent potential

    double vd = stod(argv[3]);
    double vg = stod(argv[4]);

    device dev("test", ntype, voltage<3>{ 0, vd, vg });
    dev.steady_state();

    plot(make_pair(dev.p.x, dev.phi[0].data));
    potential::plot2D(dev.p, { 0, vd, vg }, dev.n[0]);
}

static inline void den(char ** argv) {

    double vd = stod(argv[3]);
    double vg = stod(argv[4]);

    device dev("test", ntype, voltage<3>{ 0, vd, vg });
    dev.steady_state();

    plot(make_pair(dev.p.x, dev.n[0].total));
}

static inline void gstep(char ** argv) {
    // time-dependent simulation with step-signal on the gate

    double V0    = stod(argv[3]);
    double V1    = stod(argv[4]);
    double Vd    = stod(argv[5]);
    double beg   = stod(argv[6]);
    double len   = stod(argv[7]);
    double cool  = stod(argv[8]);

    stringstream ss;
    ss << "gate_step_signal/" << "V0=" << V0 << "V1=" << V1;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    signal<3> pre   = linear_signal<3>(beg,  { 0, Vd, V0 }, { 0, Vd, V0 }); // before
    signal<3> slope = linear_signal<3>(len,  { 0, Vd, V0 }, { 0, Vd, V1 }); // while
    signal<3> after = linear_signal<3>(cool, { 0, Vd, V1 }, { 0, Vd, V1 }); // after

    signal<3> sig = pre + slope + after; // complete signal

    device d("nfet", ntype, { 0, Vd, V0 });
    d.steady_state();
    d.init_time_evolution(sig.N_t);

    // get energy indices around fermi energy and init movie
    std::vector<std::pair<int, int>> E_ind1 = movie::around_Ef(d, +0.1);
    std::vector<std::pair<int, int>> E_ind2 = movie::around_Ef(d, +0.4);
    std::vector<std::pair<int, int>> E_ind;
    E_ind.reserve(E_ind1.size() + E_ind2.size());
    E_ind.insert(E_ind.end(), E_ind1.begin(), E_ind1.end());
    E_ind.insert(E_ind.end(), E_ind2.begin(), E_ind2.end());

    //movie argo(d, E_ind);

    // set voltages
    // (note: only V_g is needed, but I wanted to try it for later...)
    for (int i = 1; i < sig.N_t; ++i) {
        for (int term : {S, D, G}) {
            d.contacts[G]->V = sig.V[i][term];
        }
        d.time_step();
    }
    d.save();
}

static inline void gsquare(char ** argv) {

    double V0    = stod(argv[3]);
    double V1    = stod(argv[4]);
    double Vd    = stod(argv[5]);
    double rise  = stod(argv[6]);
    double fall  = stod(argv[7]); // after begin
    double f     = stod(argv[8]);
    double T     = stod(argv[9]);


    stringstream ss;
    ss << "gate_square_signal/" << "f=" << f;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    signal<3> sig = square_signal<3>(T, { 0, V0, Vd }, { 0, V1, Vd}, f, rise, fall);

    vec vs(sig.N_t);
    vec vd(sig.N_t);
    vec vg(sig.N_t);
    for (int i = 0; i < sig.N_t; ++i) {
        vs(i) = sig.V[i][S];
        vd(i) = sig.V[i][D];
        vg(i) = sig.V[i][G];
    }
    plot(vs, vd, vg);

    device d("nfet", ntype, sig.V[0]);
    d.steady_state();
    d.init_time_evolution(sig.N_t);

//    // get energy indices around fermi energy and init movie
//    std::vector<std::pair<int, int>> E_ind1 = movie::around_Ef(d, -0.05);
//    std::vector<std::pair<int, int>> E_ind2 = movie::around_Ef(d, -0.1);
//    std::vector<std::pair<int, int>> E_ind;
//    E_ind.reserve(E_ind1.size() + E_ind2.size());
//    E_ind.insert(E_ind.end(), E_ind1.begin(), E_ind1.end());
//    E_ind.insert(E_ind.end(), E_ind2.begin(), E_ind2.end());

//    //movie argo(d, E_ind);

    // set voltages
    for (int i = 1; i < sig.N_t; ++i) {
        for (int term : {S, D, G}) {
            d.contacts[G]->V = sig.V[i][term];
        }
        d.time_step();
    }
    d.save();
}

static inline void gsine(char ** argv) {
    // time-dependent simulation with step-signal on the gate

    double V0    = stod(argv[3]);
    double Vamp  = stod(argv[4]);
    double Vd    = stod(argv[5]);
    double beg   = stod(argv[6]);
    double len   = stod(argv[7]); // after begin
    double f     = stod(argv[8]);
    double ph    = stod(argv[9]) * 2 * M_PI; // phase

    double Vstart = V0 + Vamp * sin(ph);

    stringstream ss;
    ss << "gate_sine_signal/" << "f=" << f;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    signal<3> pre   = linear_signal<3>(beg,  { 0, Vd, Vstart }, { 0, Vd, Vstart }); // before
    signal<3> sine  =   sine_signal<3>(len,  { 0, Vd, V0 }, { 0,  0, Vamp }, f, ph); // oscillation

    signal<3> sig = pre + sine; // complete signal

//    vec vs(sig.N_t);
//    vec vd(sig.N_t);
//    vec vg(sig.N_t);
//    for (int i = 0; i < sig.N_t; ++i) {
//        vs(i) = sig.V[i][S];
//        vd(i) = sig.V[i][D];
//        vg(i) = sig.V[i][G];
//    }
//    plot(vs, vd, vg);

    device d("nfet", ntype, sig.V[0]);
    d.steady_state();
    d.init_time_evolution(sig.N_t);

    // get energy indices around fermi energy and init movie
    std::vector<std::pair<int, int>> E_ind1 = movie::around_Ef(d, -0.05);
    std::vector<std::pair<int, int>> E_ind2 = movie::around_Ef(d, -0.1);
    std::vector<std::pair<int, int>> E_ind;
    E_ind.reserve(E_ind1.size() + E_ind2.size());
    E_ind.insert(E_ind.end(), E_ind1.begin(), E_ind1.end());
    E_ind.insert(E_ind.end(), E_ind2.begin(), E_ind2.end());

    //movie argo(d, E_ind);

    // set voltages
    for (int i = 1; i < sig.N_t; ++i) {
        for (int term : {S, D, G}) {
            d.contacts[G]->V = sig.V[i][term];
        }
        d.time_step();
    }
    d.save();
}

static inline void inv_square(char ** argv) {
    double f = stod(argv[3]);
    int N = stoi(argv[4]);
    double Vs = stod(argv[5]);
    double Vd = stod(argv[6]);
    double Vg0 = stod(argv[7]);
    double Vg1 = stod(argv[8]);
    int ramp = stoi(argv[9]);

    auto s = square_signal<3>(N / f, {Vs, Vd, Vg0}, {Vs, Vd, Vg1}, f, ramp * c::dt, ramp * c::dt);

    // spam signal to cout
    for (int i = 0; i < s.N_t; ++i) {
        cout << s.V[i][G] << endl;
    }
    cout << endl << endl;

    inverter inv(nfet, pfet, 5e-17);
    inv.time_evolution(s);
}

static inline void test(char **) {
    arma::vec R0 = potential::get_R0(nfet, {0.0, 0.0, 0.0});
    charge_density n1, n2;
    n1.total = arma::vec(nfet.N_x);
    n2.total = arma::vec(nfet.N_x);

    for (int i = 0; i < nfet.N_x; ++i) {
        n1.total[i] = +5e-21 * (std::tanh((i - 10.0)/5.0) + 1) * (std::tanh(-(i + 10.0 - nfet.N_x)/5.0) + 1) / 4;
        n2.total[i] = -5e-21 * (std::tanh((i - 10.0)/5.0) + 1) * (std::tanh(-(i + 10.0 - nfet.N_x)/5.0) + 1) / 4;
    }

    std::cout << n1.total[nfet.N_x / 2] << std::endl;

    plot(n1.total);

    potential phi0(nfet, R0);
    potential phi1(nfet, R0, n1);
    potential phi2(nfet, R0, n2);

    plot(phi0.data);
    plot(phi1.data);
    plot(phi2.data);
}

int main(int argc, char ** argv) {
    setup();

    // first argument is always the number of threads
    // (can not be more than the number specified when compiling openBLAS)
    if (argc > 1) {
        omp_set_num_threads(stoi(argv[1]));
    }

    string stype = "";
    if (argc > 2) {
        stype = argv[2];
    }

    // second argument chooses the type of simulation
    if (stype == "point" && argc == 6) {
        point(argv);
    } else if (stype == "trans" && argc == 9) {
        trans(argv);
    } else if (stype == "outp" && argc == 8) {
        outp(argv);
    } else if (stype == "inv" && argc == 7) {
        inv(argv);
    } else if (stype == "ldos" && argc == 8) {
        ldos(argv);
    } else if (stype == "pot" && argc == 5) {
        pot(argv);
    } else if (stype == "den" && argc == 5) {
        den(argv);
    } else if (stype == "ro" && argc == 6) {
        ro(argv);
    } else if (stype == "gstep" && argc == 9) {
        gstep(argv);
    } else if (stype == "gsine" && argc == 10) {
        gsine(argv);
    } else if (stype == "gsquare" && argc == 10) {
        gsquare(argv);
    } else if (stype == "inv_square" && argc == 10) {
        inv_square(argv);
    } else {
        test(argv);
    }

    return 0;
}*/
