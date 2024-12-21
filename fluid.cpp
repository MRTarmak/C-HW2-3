#include <iostream>
#include <array>
#include <utility>
#include <ranges>
#include <cassert>
#include <algorithm>
#include <random>
#include <fstream>
#include <chrono>

using namespace std;

constexpr size_t T = 1'000;
constexpr std::array<pair<int, int>, 4> deltas{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

template <typename T>
class Matrix {
public:
    Matrix() = default;
    Matrix(size_t n, size_t m) : data(n, std::vector<T>(m, T{})), n(n), m(m) {};
    Matrix(const Matrix& other) = default;
    Matrix& operator= (const Matrix& other) = default;

    ~Matrix() {
        n = 0LU;
        m = 0LU;
    }

    void clean() {
        for (auto i = 0LU; i < n; i++)
            for (auto j = 0LU; j < m; j++)
                data[i][j] = T{};
    }

    T* operator[] (size_t index) {
        return &data[index][0];
    }

    size_t get_n() const {return n;}
    size_t get_m() const {return m;}

private:
    std::vector<std::vector<T>> data;

    size_t n = 0;
    size_t m = 0;
};

size_t N, M;
Matrix<char> field;

template<unsigned N, bool is_fast>
struct number_type {
    using type = std::conditional_t<N <= 8, int_fast8_t,
            std::conditional_t<N <= 16, int_fast16_t,
                    std::conditional_t<N <= 32, int_fast32_t,
                            std::conditional_t<N <= 64, int_fast64_t, void>>>>;
};

template<unsigned N>
struct number_type<N, false> {
    using type = std::conditional_t<N == 8, int8_t,
            std::conditional_t<N == 16, int16_t,
                    std::conditional_t<N == 32, int32_t,
                            std::conditional_t<N == 64, int64_t, void>>>>;
};

template<unsigned N, bool is_fast>
using num_type = number_type<N, is_fast>::type;

template<unsigned N, unsigned K, bool fast = false>
struct Fixed {
    using num_t = num_type<N, fast>;
    static const int n = N;
    static const int k = K;

    num_t v;

    template<int N1, int K1, bool is_fast1>
    constexpr Fixed(const Fixed<N1, K1, is_fast1> &other) {
        if constexpr (K1 > K)
            v = other.v >> (K1 - K);
        else
            v = other.v << (K - K1);
    }

    constexpr Fixed(int8_t v) : v(v << K) {}
    constexpr Fixed(int16_t v) : v(v << K) {}
    constexpr Fixed(int32_t v) : v(v << K) {}
    constexpr Fixed(int64_t v) : v(v << K) {}
    constexpr Fixed(float f) : v(f * (1LL << K)) {}
    constexpr Fixed(double f) : v(f * (1LL << K)) {}
    constexpr Fixed() : v(0) {}

    explicit constexpr operator float() { return float(v) / (1LL << K); }
    explicit constexpr operator double() { return double(v) / (1LL << K); }

    static constexpr Fixed from_raw(int x) {
        Fixed ret{};
        ret.v = x;
        return ret;
    }

    auto operator<=>(const Fixed &) const = default;
    bool operator==(const Fixed &) const = default;

    friend Fixed operator+(Fixed a, Fixed b) {
        return Fixed::from_raw(a.v + b.v);
    }

    friend Fixed operator-(Fixed a, Fixed b) {
        return Fixed::from_raw(a.v - b.v);
    }

    friend Fixed operator*(Fixed a, Fixed b) {
        return Fixed::from_raw(((int64_t) a.v * b.v) >> K);
    }

    friend Fixed operator/(Fixed a, Fixed b) {
        return Fixed::from_raw(((int64_t) a.v << K) / b.v);
    }

    friend Fixed &operator+=(Fixed &a, Fixed b) {
        return a = a + b;
    }

    friend Fixed &operator-=(Fixed &a, Fixed b) {
        return a = a - b;
    }

    friend Fixed &operator*=(Fixed &a, Fixed b) {
        return a = a * b;
    }

    friend Fixed &operator/=(Fixed &a, Fixed b) {
        return a = a / b;
    }

    friend Fixed operator-(Fixed x) {
        return Fixed::from_raw(-x.v);
    }

    friend Fixed fabs(Fixed x) {
        if (x.v < 0) {
            x.v = -x.v;
        }
        return x;
    }

    friend std::ostream &operator<<(std::ostream &out, Fixed x) {
        return out << x.v / (double) (1 << K);
    }
};

Fixed<32, 16> rho[256];

Matrix<Fixed<32, 16>> p, old_p;

struct VectorField {
    Matrix<array<Fixed<32, 16>, deltas.size()>> v;
    Fixed<32, 16> &add(int x, int y, int dx, int dy, Fixed<32, 16> dv) {
        return get(x, y, dx, dy) += dv;
    }

    Fixed<32, 16> &get(int x, int y, int dx, int dy) {
//        size_t i = ranges::find(deltas, pair(dx, dy)) - deltas.begin();
//        assert(i < deltas.size());
//        return v[x][y][i];

        switch (4*dx + dy)
        {
            case -1: return v[x][y][2];
            case  1: return v[x][y][3];
            case -4: return v[x][y][0];
            case  4: return v[x][y][1];
        }
    }
};

VectorField velocity{}, velocity_flow{};
Matrix<int> last_use;
int UT = 0;

mt19937 rnd(1337);

tuple<Fixed<32, 16>, bool, pair<int, int>> propagate_flow(int x, int y, Fixed<32, 16> lim) {
    last_use[x][y] = UT - 1;
    Fixed<32, 16> ret = 0.0;
    for (auto [dx, dy] : deltas) {
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] != '#' && last_use[nx][ny] < UT) {
            auto cap = velocity.get(x, y, dx, dy);
            auto flow = velocity_flow.get(x, y, dx, dy);
            if (flow == cap) {
                continue;
            }
            // assert(v >= velocity_flow.get(x, y, dx, dy));
            auto vp = min(lim, cap - flow);

            ////////////////////////////
            if (vp < 0.001)
                continue;
            ////////////////////////////

            if (last_use[nx][ny] == UT - 1) {
                velocity_flow.add(x, y, dx, dy, vp);
                last_use[x][y] = UT;
                // cerr << x << " " << y << " -> " << nx << " " << ny << " " << vp << " / " << lim << "\n";
                return {vp, 1, {nx, ny}};
            }
            auto [t, prop, end] = propagate_flow(nx, ny, vp);
            ret += t;
            if (prop) {
                velocity_flow.add(x, y, dx, dy, t);
                last_use[x][y] = UT;
                // cerr << x << " " << y << " -> " << nx << " " << ny << " " << t << " / " << lim << "\n";
                return {t, prop && end != pair(x, y), end};
            }
        }
    }
    last_use[x][y] = UT;
    return {ret, 0, {0, 0}};
}

Fixed<32, 16> random01() {
    return Fixed<32, 16>::from_raw((rnd() & ((1 << 16) - 1)));
}

void propagate_stop(int x, int y, bool force = false) {
    if (!force) {
        bool stop = true;
        for (auto [dx, dy] : deltas) {
            int nx = x + dx, ny = y + dy;
            if (field[nx][ny] != '#' && last_use[nx][ny] < UT - 1 && velocity.get(x, y, dx, dy) > 0.0) {
                stop = false;
                break;
            }
        }
        if (!stop) {
            return;
        }
    }
    last_use[x][y] = UT;
    for (auto [dx, dy] : deltas) {
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] == '#' || last_use[nx][ny] == UT || velocity.get(x, y, dx, dy) > 0.0) {
            continue;
        }
        propagate_stop(nx, ny);
    }
}

Fixed<32, 16> move_prob(int x, int y) {
    Fixed<32, 16> sum = 0.0;
    for (size_t i = 0; i < deltas.size(); ++i) {
        auto [dx, dy] = deltas[i];
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] == '#' || last_use[nx][ny] == UT) {
            continue;
        }
        auto v = velocity.get(x, y, dx, dy);
        if (v < 0.0) {
            continue;
        }
        sum += v;
    }
    return sum;
}

struct ParticleParams {
    char type;
    Fixed<32, 16> cur_p;
    array<Fixed<32, 16>, deltas.size()> v;

    void swap_with(int x, int y) {
        swap(field[x][y], type);
        swap(p[x][y], cur_p);
        swap(velocity.v[x][y], v);
    }
};

bool propagate_move(int x, int y, bool is_first) {
    last_use[x][y] = UT - is_first;
    bool ret = false;
    int nx = -1, ny = -1;
    do {
        std::array<Fixed<32, 16>, deltas.size()> tres;
        Fixed<32, 16> sum = 0.0;
        for (size_t i = 0; i < deltas.size(); ++i) {
            auto [dx, dy] = deltas[i];
            int nx = x + dx, ny = y + dy;
            if (field[nx][ny] == '#' || last_use[nx][ny] == UT) {
                tres[i] = sum;
                continue;
            }
            auto v = velocity.get(x, y, dx, dy);
            if (v < 0.0) {
                tres[i] = sum;
                continue;
            }
            sum += v;
            tres[i] = sum;
        }

        if (sum == 0.0) {
            break;
        }

        Fixed p = random01() * sum;
        size_t d = std::ranges::upper_bound(tres, p) - tres.begin();

        auto [dx, dy] = deltas[d];
        nx = x + dx;
        ny = y + dy;
        assert(velocity.get(x, y, dx, dy) > 0.0 && field[nx][ny] != '#' && last_use[nx][ny] < UT);

        ret = (last_use[nx][ny] == UT - 1 || propagate_move(nx, ny, false));
    } while (!ret);
    last_use[x][y] = UT;
    for (size_t i = 0; i < deltas.size(); ++i) {
        auto [dx, dy] = deltas[i];
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] != '#' && last_use[nx][ny] < UT - 1 && velocity.get(x, y, dx, dy) < 0.0) {
            propagate_stop(nx, ny);
        }
    }
    if (ret) {
        if (!is_first) {
            ParticleParams pp{};
            pp.swap_with(x, y);
            pp.swap_with(nx, ny);
            pp.swap_with(x, y);
        }
    }
    return ret;
}

Matrix<int64_t> dirs;

int main() {
    double r1, r2;
    double g0;

    auto start = std::chrono::high_resolution_clock::now();

    std::string line;
    std::ifstream input("input1.txt");
    if (input.is_open()) {
        input >> N >> M >> r1 >> r2 >> g0;

        field = Matrix<char>(N, M);

        std::getline(input, line);
        for (int i = 0; i < N; i++) {
            std::getline(input, line);

            for (int j = 0; j < M; j++)
            {
                field[i][j] = line[j];
            }
        }
    }
    input.close();

    Fixed<32, 16> g = g0;
    rho[' '] = r1;
    rho['.'] = r2;
    p = Matrix<Fixed<32, 16>>(N, M);
    old_p = Matrix<Fixed<32, 16>>(N, M);
    dirs = Matrix<int64_t >(N, M);
    last_use = Matrix<int>(N, M);
    velocity.v = Matrix<array<Fixed<32, 16>, deltas.size()>>(N, M);
    velocity_flow.v = Matrix<array<Fixed<32, 16>, deltas.size()>>(N, M);

    for (size_t x = 0; x < N; ++x) {
        for (size_t y = 0; y < M; ++y) {
            if (field[x][y] == '#')
                continue;
            for (auto [dx, dy] : deltas) {
                dirs[x][y] += (field[x + dx][y + dy] != '#');
            }
        }
    }

    for (size_t i = 0; i < T; ++i) {
        
        Fixed<32, 16> total_delta_p = 0.0;
        // Apply external forces
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] == '#')
                    continue;
                if (field[x + 1][y] != '#')
                    velocity.add(x, y, 1, 0, g);
            }
        }

        // Apply forces from p
        old_p = p;
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] == '#')
                    continue;
                for (auto [dx, dy] : deltas) {
                    int nx = x + dx, ny = y + dy;
                    if (field[nx][ny] != '#' && old_p[nx][ny] < old_p[x][y]) {
                        auto delta_p = old_p[x][y] - old_p[nx][ny];
                        auto force = delta_p;
                        auto &contr = velocity.get(nx, ny, -dx, -dy);
                        if (contr * rho[(int) field[nx][ny]] >= force) {
                            contr -= force / rho[(int) field[nx][ny]];
                            continue;
                        }
                        force -= contr * rho[(int) field[nx][ny]];
                        contr = 0.0;
                        velocity.add(x, y, dx, dy, force / rho[(int) field[x][y]]);
                        p[x][y] -= force / dirs[x][y];
                        total_delta_p -= force / dirs[x][y];
                    }
                }
            }
        }

        // Make flow from velocities
        velocity_flow.v.clean();
        bool prop = false;
        do {
            UT += 2;
            prop = 0;
            for (size_t x = 0; x < N; ++x) {
                for (size_t y = 0; y < M; ++y) {
                    if (field[x][y] != '#' && last_use[x][y] != UT) {
                        auto [t, local_prop, _] = propagate_flow(x, y, 1.0);
                        if (t > 0.0) {
                            prop = 1;
                        }
                    }
                }
            }
        } while (prop);

        // Recalculate p with kinetic energy
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] == '#')
                    continue;
                for (auto [dx, dy] : deltas) {
                    auto old_v = velocity.get(x, y, dx, dy);
                    auto new_v = velocity_flow.get(x, y, dx, dy);
                    if (old_v > 0.0) {
                        assert(new_v <= old_v);
                        velocity.get(x, y, dx, dy) = new_v;
                        auto force = (old_v - new_v) * rho[(int) field[x][y]];
                        if (field[x][y] == '.')
                            force *= 0.8;
                        if (field[x + dx][y + dy] == '#') {
                            p[x][y] += force / dirs[x][y];
                            total_delta_p += force / dirs[x][y];
                        } else {
                            p[x + dx][y + dy] += force / dirs[x + dx][y + dy];
                            total_delta_p += force / dirs[x + dx][y + dy];
                        }
                    }
                }
            }
        }

        UT += 2;
        prop = false;
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] != '#' && last_use[x][y] != UT) {
                    if (random01() < move_prob(x, y)) {
                        prop = true;
                        propagate_move(x, y, true);
                    } else {
                        propagate_stop(x, y, true);
                    }
                }
            }
        }

        if (prop) {
            cout << "Tick " << i << ":\n";
            for (size_t x = 0; x < N; ++x) {
                cout << field[x] << "\n";
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Время выполнения: " << duration.count() << " секунд" << std::endl;
}