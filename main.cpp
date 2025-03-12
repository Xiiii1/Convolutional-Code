#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <bitset>
#include <iomanip>
#include <algorithm> 
#include <map>
#include <climits>
#include <limits>
#include <set>
#include <time.h>

using namespace std;

void Parser(ifstream& input, string& N_str, string& L_str, string& SNR_str, string& Seed_str, string& decision_str, int& N, int& L, double& SNR, int& Seed, int& decision);
void getU(const int N, vector<int>& u);
void Encoder(const int N, const vector<int>& u, vector<int>& c);
void Modulator(const vector<int>& c, vector<int>& x);
double ran1(vector<long>& idum);
void normal(double& n1, double& n2, double sigma, vector<long>& idum);
void AWGN(vector<double>& y1, vector<double>& y2, const vector<int>& x, double n1, double n2, const double sigma, vector<long>& idum);
int TrellisOutput(int shiftRegister, const vector<int>& generator);
void generateTrellisTable(vector<vector<int>>& trellisTable);
double calDistance(double received1, double received2, double output1, double output2);
int reverseBits(int v);
void ACS(vector<vector<int>>& paths, vector<double>& acc_metric);
void decoder(int N, const int& L, const vector<double>& y1, const vector<double>& y2, vector<double>& acc_metric, vector<vector<int>>& paths, const vector<vector<int>>& trellisTable, vector<int>& u_est, double variance, int decision);

const int m = 6;  // the number of registers
const int num_states = 1 << m; // the number of states (2^m)
vector<int> shiftRegister_enc(m + 1, 0); // 6-stage shift register, see u(i) as s0(i)
vector<vector<int>> G = {
    { 1,0,1,1,0,1,1 },    // 1 + D^2 + D^3 + D^5 + D^6
    { 1,1,1,1,0,0,1 }    // 1 + D + D^2 + D^3 + D^6
};

/* AWGN part */
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

int main()
{
    // Start the timer to measure the execution time of the simulation
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    /* (2,1,6) conovolutional code */

    ifstream inputFile("Sim.txt");
    if (!inputFile.is_open()) {
        cerr << "Error opening the input file..." << endl;
        return 1;
    }
    string N_str, L_str, SNR_str, Seed_str, decision_str;
    int N, L, Seed, decision;
    double SNR;
    Parser(inputFile, N_str, L_str, SNR_str, Seed_str, decision_str, N, L, SNR, Seed, decision);

    vector<int> u;
    getU(N, u);

    vector<int> c;
    Encoder(N, u, c);

    vector<int> x;  //x=(-1)^c
    Modulator(c, x);

    vector<long> idum = { Seed };
    double sigma = sqrt(1.0 / pow(10.0, SNR / 10.0));       // R=1/2 (#input/#output)
    double n1, n2;
    vector<double> y1, y2;
    AWGN(y1, y2, x, n1, n2, sigma, idum);

    vector<vector<int>> trellisTable;   // trellisTable store the information, [states next state input ouput(y1) output(y2)]
    generateTrellisTable(trellisTable);

    vector<double> hard_output1, hard_output2;
    vector<int> u_est;
    vector<double> acc_metric;
    vector<vector<int>> paths;               // store ith state when we choose it.
    ofstream outputFile;
    if (!decision) {
        outputFile.open("Output_Hard.txt");
        /* demodulate y to {0,1} */
        // decision_output = hard_output1 hard_output2 hard_output1 hard_output2...
        for (int i = 0;i < y1.size();i++) {
            hard_output1.emplace_back((y1[i] >= 0) ? 0 : 1);
            hard_output2.emplace_back((y2[i] >= 0) ? 0 : 1);
        }
        for (int i = 0;i < hard_output1.size();i++) {
            outputFile << hard_output1[i] << " ";
            outputFile << hard_output2[i] << " ";
        }
        outputFile << "%hard-decision output: 2*(N+m) elements" << endl;
        decoder(N, L, hard_output1, hard_output2, acc_metric, paths, trellisTable, u_est, sigma * sigma, decision);
        for (int i = 0;i < u_est.size();i++) {
            outputFile << u_est[i] << " ";
        }
        outputFile << "%decoded information bits";
    }
    else if (decision) {
        outputFile.open("Output_Soft.txt");
        for (int i = 0;i < y1.size();i++) {
            outputFile << fixed << setprecision(6) << y1[i] << " ";
            outputFile << fixed << setprecision(6) << y2[i] << " ";
        }
        outputFile << "%y: 2*(N+m) elements" << endl;
        decoder(N, L, y1, y2, acc_metric, paths, trellisTable, u_est, sigma * sigma, decision);
        for (int i = 0;i < u_est.size();i++) {
            outputFile << u_est[i] << " ";
        }
        outputFile << "%decoded information bits";
    }

    if (!outputFile.is_open()) {
        cerr << "Error opening the output file..." << endl;
        return 1;
    }

    // BER
    cout << "Decision type :" << ((decision) ? " soft\n" : " hard\n");
    cout << "N = " << N << endl;
    //cout << "size of u_est = " << u_est.size() << endl;
    int numError = 0;
    for (int i = 0;i < u.size();i++) {
        if (u[i] != u_est[i]) {
            numError++;
            //cout << "error @ " << i << endl;
        }
    }
    cout << "#decoded bit errors = " << numError << endl;
    double BER = static_cast<double>(numError) / u.size();
    cout << "BER = " << BER << " @ SNR = " << SNR << "dB" << endl;
    cout << "-----------------------------\n";
    /* close the files */
    inputFile.close();
    outputFile.close();

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("running time: %f sec\n", cpu_time_used);
    system("pause");
    return 0;
}

void Parser(ifstream& input, string& N_str, string& L_str, string& SNR_str, string& Seed_str, string& decision_str, int& N, int& L, double& SNR, int& Seed, int& decision) {
    getline(input, N_str);
    N_str = N_str.substr(0, N_str.find('%'));
    N = stoi(N_str);
    getline(input, L_str);
    L_str = L_str.substr(0, L_str.find('%'));
    L = stoi(L_str);
    getline(input, SNR_str);
    SNR_str = SNR_str.substr(0, SNR_str.find('%'));
    SNR = stod(SNR_str);
    getline(input, Seed_str);
    Seed_str = Seed_str.substr(0, Seed_str.find('%'));
    Seed = stoi(Seed_str);
    getline(input, decision_str);
    decision_str = decision_str.substr(0, decision_str.find('%'));
    decision = stoi(decision_str);
}

void getU(const int N, vector<int>& u) {
    u.resize(N);
    if (N > 6) {
        u[0] = 1;
        u[1] = u[2] = u[3] = u[4] = u[5] = 0;
        for (int i = 6;i < N;i++) { u[i] = u[i - 5] ^ u[i - 6]; }
    }
    else {
        u[0] = 1;
        for (int i = 1;i < N;i++) { u[i] = 0; }
    }
}

void Encoder(const int N, const vector<int>& u, vector<int>& c) {
    for (int i = 0;i < N;i++) {
        // shift register update
        shiftRegister_enc.insert(shiftRegister_enc.begin(), u[i]);
        shiftRegister_enc.pop_back();

        // calculate output bits
        int out1 = 0, out2 = 0;
        for (int j = 0; j <= m; j++) {
            out1 = out1 ^ (shiftRegister_enc[j] * G[0][j]);
            out2 = out2 ^ (shiftRegister_enc[j] * G[1][j]);
        }
        // store output bits
        c.emplace_back(out1);
        c.emplace_back(out2);
    }

    // after truncation length
    for (int i = 0; i < m; i++) {
        // input = 0
        shiftRegister_enc.insert(shiftRegister_enc.begin(), 0);
        shiftRegister_enc.pop_back();

        int out1 = 0, out2 = 0;
        for (int j = 0; j <= m; j++) {
            out1 = out1 ^ (shiftRegister_enc[j] * G[0][j]);
            out2 = out2 ^ (shiftRegister_enc[j] * G[1][j]);
        }
        c.emplace_back(out1);
        c.emplace_back(out2);
    }
}

void Modulator(const vector<int>& c, vector<int>& x) {
    for (int i = 0;i < c.size();i++) {
        x.emplace_back((c[i] == 0) ? 1 : -1);
    }
}

double ran1(vector<long>& idum) {
    int j;
    long k;
    static long iy = 0;
    static vector<long> iv(NTAB);
    double temp;

    if (idum[0] <= 0 || iy == 0) {
        if (-idum[0] < 1)
            idum[0] = 1;
        else
            idum[0] = -idum[0];

        for (j = NTAB + 7; j >= 0; j--) {
            k = idum[0] / IQ;
            idum[0] = IA * (idum[0] - k * IQ) - IR * k;
            if (idum[0] < 0)
                idum[0] += IM;
            if (j < NTAB)
                iv[j] = idum[0];
        }
        iy = iv[0];
    }

    k = idum[0] / IQ;
    idum[0] = IA * (idum[0] - k * IQ) - IR * k;
    if (idum[0] < 0)
        idum[0] += IM;

    j = iy / NDIV;
    iy = iv[j];
    iv[j] = idum[0];

    if ((temp = AM * iy) > RNMX)
        return RNMX;
    else
        return temp;
}

void normal(double& n1, double& n2, double sigma, vector<long>& idum) {
    double temp1, temp2, s;
    do {
        temp1 = ran1(idum);
        temp2 = ran1(idum);
        temp1 = 2 * temp1 - 1;
        temp2 = 2 * temp2 - 1;
        s = temp1 * temp1 + temp2 * temp2;
    } while (s >= 1.0);
    n1 = sigma * temp1 * sqrt((-2.0) * log(s) / s);
    n2 = sigma * temp2 * sqrt((-2.0) * log(s) / s); // log ≡ ln in c++
}

void AWGN(vector<double>& y1, vector<double>& y2, const vector<int>& x, double n1, double n2, const double sigma, vector<long>& idum) {
    for (int i = 0;i < x.size();i += 2) {
        normal(n1, n2, sigma, idum);
        y1.emplace_back(x[i] + n1);
        y2.emplace_back(x[i + 1] + n2);
    }
}

int TrellisOutput(int shiftRegister, const vector<int>& generator) {
    int output = 0;
    for (int i = 0; i < generator.size(); i++) {
        if (generator[i]) {
            output ^= (shiftRegister >> i) & 1; // XOR operation for binary addition
        }
    }
    return output;
}

int reverseBits(int v) {
    int rev_v = 0;
    for (int i = 0; i < m; i++) {
        rev_v = (rev_v << 1) | (v & 1); // extract the LSB of v and insert to rev_v leftly
        v >>= 1;  // right-shifting, remove the rightest bit
    }
    return rev_v;
}

void generateTrellisTable(vector<vector<int>>& trellisTable) {
    // Iterate over all states and inputs
    for (int state = 0; state < num_states; state++) {
        for (int input = 0; input <= 1; input++) {
            // Update shift register to include the new input
            int shift_register = ((state << 1) | input) & 0x7F;  // keep 7 bits (input seem as s0)

            // Calculate next state (keep only the lower m bits)
            int next_state = shift_register & (num_states - 1);

            // Reverse the state and next_state
            int reversed_state = reverseBits(state);
            int reversed_next_state = reverseBits(next_state);

            // Calculate outputs
            int output1 = TrellisOutput(shift_register, G[0]);
            int output2 = TrellisOutput(shift_register, G[1]);

            // Store the information in the trellis table
            trellisTable.emplace_back(vector<int>{ state, reversed_state, next_state, reversed_next_state, input, output1, output2 });
        }
    }
}

double calDistance(double received1, double received2, double output1, double output2) {
    return (received1 - output1) * (received1 - output1) + (received2 - output2) * (received2 - output2);
}

void ACS(vector<vector<int>>& paths, vector<double>& acc_metric) {
    map<int, pair<int, double>> state_map; // <key: next_state, value: <path_index, metric>>
    set<int> index_to_remove;   // target the index of worse path with increasing order

    for (int i = 0; i < paths.size(); i++) {
        int next_state = paths[i].back();
        if (state_map.find(next_state) != state_map.end()) {
            // if state is exist, compare the metric
            int existing_pathIndex = state_map[next_state].first;
            int existing_metric = state_map[next_state].second;

            if (acc_metric[i] < existing_metric) {
                // update next state
                index_to_remove.insert(existing_pathIndex);
                state_map[next_state] = { i, acc_metric[i] };
            }
            else if (acc_metric[i] > existing_metric) {
                index_to_remove.insert(i);
            }
            // when same metric, choose the path with smaller reversed_current_state
            else {
                int current_state1 = paths[i][paths[i].size() - 2];  // current state
                int current_state2 = paths[existing_pathIndex][paths[existing_pathIndex].size() - 2];
                int rev_current_state1 = reverseBits(current_state1);
                int rev_current_state2 = reverseBits(current_state2);

                if (rev_current_state1 > rev_current_state2) {
                    index_to_remove.insert(i);
                }
                else {
                    // update next state
                    index_to_remove.insert(existing_pathIndex);
                    state_map[next_state] = { i, acc_metric[i] };
                }
            }
        }
        else {
            // update next state
            state_map[next_state] = { i, acc_metric[i] };
        }
    }

    // remove the targetd path and corresponding acc_metric (from back)
    for (auto it = index_to_remove.rbegin(); it != index_to_remove.rend(); it++) {
        int idx = *it;
        paths.erase(paths.begin() + idx);
        acc_metric.erase(acc_metric.begin() + idx);
    }
}

void decoder(int N, const int& L, const vector<double>& y1, const vector<double>& y2, vector<double>& acc_metric, vector<vector<int>>& paths, const vector<vector<int>>& trellisTable, vector<int>& u_est, double variance, int decision) {
    // start from state 000000
    bool initial = true;
    paths.resize(num_states, vector<int>{0});
    acc_metric.resize(num_states, 0.0);
    // each y_index represents the survivor of trellis diagram between t and t+1
    for (int y_index = 0;y_index < y1.size();y_index++) {
        vector<vector<int>> next_paths(2 * num_states);
        vector<double> next_metric(2 * num_states, 0.0);

        if (initial) {
            int group_size = num_states / (1 << (y_index)); // calculate group size before going to the next state (2^(m-y_index))
            for (int group_start = 0; group_start < num_states; group_start += group_size) {
                for (int input = 0; input <= 1; input++) {
                    for (int i = 0; i < group_size / 2; i++) {
                        // Calculate current and next index
                        int current_index = group_start + i;
                        int next_index = group_start + i + (input * group_size / 2);
                        // Get current and next state
                        int current_state = paths[current_index].back(); // reverse order
                        //int next_state = ((input << (m - 1)) | (current_state >> 1));   // reverse order
                        int next_state = reverseBits(((reverseBits(current_state) << 1) & 0x3F | input));  // reverse order
                        double out1, out2;
                        // Find corresponding outputs in the trellis table
                        for (const auto& row : trellisTable) {
                            if (row[1] == current_state && row[3] == next_state) {
                                if (decision) {
                                    out1 = ((row[5] == 0) ? 1.0 : -1.0);
                                    out2 = ((row[6] == 0) ? 1.0 : -1.0);
                                    break;
                                }
                                else {
                                    out1 = row[5];
                                    out2 = row[6];
                                    break;
                                }
                            }
                        }
                        // update paths and acc_metric
                        next_paths[next_index] = paths[current_index];
                        next_paths[next_index].emplace_back(next_state);
                        next_metric[next_index] = acc_metric[current_index] + calDistance(y1[y_index], y2[y_index], out1, out2);
                    }
                }
            }

            // update paths and acc_metric
            paths = next_paths;
            acc_metric = next_metric;

            // from 7th input, the path merging promblem shows.
            if (y_index == 5) {
                initial = false;
                paths.resize(num_states);
                acc_metric.resize(num_states);
            }
        }
        else {
            // 64 pahts -> 128 paths, after path marging selectivity, 128 paths -> 64 paths
            for (int path_index = 0;path_index < num_states;path_index++) {
                // get current state
                int current_state = paths[path_index].back() & 0x7F;
                // get next state according to input 1 or 0
                // calculate acc_metric
                // check if the next state is duplicating, if so, using function "ACS" to choose the better path
                for (int input = 0; input <= 1; input++) {
                    //int next_state = ((input << (m - 1)) | (current_state >> 1));   // reverse order
                    int next_state = reverseBits((((reverseBits(current_state) << 1) & 0x3F) | input)) & 0x7F;  // reverse order
                    double out1, out2;
                    // compare with the trellisTable to get output bits
                    for (int table_index = 0; table_index < 2 * num_states; table_index++) {
                        if (trellisTable[table_index][1] == current_state && trellisTable[table_index][3] == next_state) {
                            if (decision) {
                                out1 = ((trellisTable[table_index][5] == 0) ? 1.0 : -1.0);
                                out2 = ((trellisTable[table_index][6] == 0) ? 1.0 : -1.0);
                                break;
                            }
                            else {
                                out1 = trellisTable[table_index][5];
                                out2 = trellisTable[table_index][6];
                                break;
                            }
                        }
                    }

                    // update paths and acc_metric
                    int next_index = 2 * path_index + input;
                    next_paths[next_index] = paths[path_index];   // add next state
                    next_paths[next_index].emplace_back(next_state);
                    if ((y_index >= (y1.size() - m) && input == 1)) {// converging part
                        next_metric[next_index] = numeric_limits<long double>::max();
                    }
                    else {
                        next_metric[next_index] = acc_metric.at(path_index) + calDistance(y1[y_index], y2[y_index], out1, out2);
                    }
                }
            }
            //select the better path
            ACS(next_paths, next_metric);
            paths = next_paths;
            acc_metric = next_metric;

            // traceback to output the tail information bit
            // best-state: select the path with smallest acc_metric (while same ACS, choose the upper path)
            // compare the state @ t=0 and the state @ t=1 with the trellisTable to get u_est
            if (y_index >= L - 1) {
                auto min_metric = min_element(acc_metric.begin(), acc_metric.end());    // find minimum acc_metric
                double min_value = *min_metric;
                bool same_min = count(acc_metric.begin(), acc_metric.end(), min_value) > 1;
                int target_path = -1;
                if (same_min) {
                    int upper_state = INT_MAX;
                    for (int i = 0; i < acc_metric.size(); i++) {
                        if (acc_metric.at(i) == min_value) {
                            int temp_state = reverseBits(paths.at(i).back());
                            if (temp_state < upper_state) {
                                upper_state = temp_state;
                                target_path = i;
                            }
                        }
                    }
                }
                else {
                    target_path = static_cast<int>(distance(acc_metric.begin(), min_metric));   // find the path index
                }
                if (target_path == -1) { cerr << "what?????" << endl; return; }
                int first_state = paths.at(target_path).at(0);
                int second_state = paths.at(target_path).at(1);
                // compare with trellisTable to get desired output
                for (int table_index = 0; table_index < 2 * num_states; table_index++) {
                    if (trellisTable[table_index][1] == first_state && trellisTable[table_index][3] == second_state) {
                        u_est.emplace_back(trellisTable[table_index][4]);
                        for (int i = 0;i < paths.size();i++) {
                            paths.at(i).erase(paths.at(i).begin());
                        }
                        break;
                    }
                }
                // for last window, output all the bits
                if (y_index == y1.size() - 1) {
                    // find the path with minimum acc_metric and input = 0 
                    for (int remaining = 0;remaining < paths.at(target_path).size() - 1;remaining++) {
                        first_state = paths.at(target_path).at(remaining);
                        second_state = paths.at(target_path).at(1 + remaining);
                        // compare with trellisTable to get output
                        for (int table_index = 0; table_index < 2 * num_states; table_index++) {
                            if (trellisTable[table_index][1] == first_state && trellisTable[table_index][3] == second_state) {
                                u_est.emplace_back(trellisTable[table_index][4]);
                                break;
                            }
                        }
                    }
                }
            }
            // renormalize
            if (N >= 10000 && decision) {
                if (y_index % L == 0) {
                    auto min_normal = min_element(acc_metric.begin(), acc_metric.end());    // find minimum acc_metric
                    double normal_value = *min_normal;
                    for (int i = 0;i < acc_metric.size();i++) {
                        acc_metric[i] -= normal_value;
                    }
                }
            }
        }
    }
}