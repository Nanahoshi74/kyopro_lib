#include <bits/stdc++.h>
#define rep(i, n) for (int i = 0; i < (int)(n); i++)
#define repi(i,a,b) for(int i = a; i <= (int)(b); i++)
#define rng(i,a,b) for(int i = a; i < (int)(b); i++)
typedef long long ll;
using namespace std;
using P = pair<ll,ll>;
#define all(a) a.begin(),a.end()

struct V{//************************幾何ライブラリ****************************
    int x,y;
    V(int x = 0, int y = 0): x(x),y(y){}
    V operator+(const V& a) const{//ベクトルの足し算を定義
        return V(x+a.x,y+a.y);
    }
    V operator-(const V& a) const{//ベクトルの引き算を定義
        return V(x-a.x,y-a.y);
    }
    V operator*(const int p) const{//ベクトルの定数倍
        return V(x*p,y*p);
    }
    int cross(const V& a) const{//外積を求める
        return x*a.y - y*a.x;
    }
    int ccw(const V& a) const{
        int area = cross(a);
        if(area > 0) return +1;//ccw(反時計回り)
        if(area < 0) return -1; //cw(時計回り)
        return 0; //collinear(1直線上)
    }
};
/*vector<V> p(4);
    rep(i,4) cin >> p[i].x >> p[i].y;

    rep(i,4){
        V A = p[i];
        V B = p[(i+1) % 4];
        V C = p[(i+2) % 4];
        V b = B-A,c = C-A;
        if(b.ccw(c) != 1){
            cout << "No" << endl;   https://atcoder.jp/contests/abc266/tasks/abc266_c
            return 0;
        }
    }*/

////////////////ベクトルの回転///////////////////
P rotate(P vec,double deg){
    P p;
    double PI = acos(-1);
    double rad = deg * (PI/180);
    p.first = vec.first * cos(rad) - vec.second * sin(rad);///cos(π/2)などを正確に整数として求めたい場合。sinとcosの前に(ll)をつける
    p.second = vec.first * sin(rad) + vec.second * cos(rad);
    return p;
}

vector<long long> divisor(long long n) {//**************************約数列挙************************:
    vector<long long> ret;
    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            ret.push_back(i);
            if (i * i != n) ret.push_back(n / i);
        }
    }
    sort(ret.begin(), ret.end()); // 昇順に並べる
    return ret;
}

vector<int> Eratosthenes( const ll N )//********************素数列挙******************
{
    vector<bool> is_prime( N + 1 );
    for( int i = 0; i <= N; i++ )
    {
        is_prime[ i ] = true;
    }
    vector<int> P;
    for( int i = 2; i <= N; i++ )
    {
        if( is_prime[ i ] )
        {
            for( int j = 2 * i; j <= N; j += i )
            {
                is_prime[ j ] = false;
            }
            P.emplace_back( i );
        }
    }
    return P;
}

bool IsPrime(ll num)//**************************素数判定***************************:
{
    if (num < 2) return false;
    else if (num == 2) return true;
    else if (num % 2 == 0) return false; //偶数はあらかじめ除く

    double sqrtNum = sqrt(num);
    for (int i = 3; i <= sqrtNum; i += 2)
    {
        if (num % i == 0)
        {
            // 素数ではない
            return false;
        }
    }

    // 素数である
    return true;
}

///////////////////////////////////////////素因数分解/////////////////////////////////////////////

vector<pair<long long, long long> > prime_factorize(long long N) {
    vector<pair<long long, long long> > res;
    for (long long a = 2; a * a <= N; ++a) {
        if (N % a != 0) continue;
        long long ex = 0; // 指数

        // 割れる限り割り続ける
        while (N % a == 0) {
            ++ex;
            N /= a;
        }

        // その結果を push
        res.push_back({a, ex});
    }

    // 最後に残った数について
    if (N != 1) res.push_back({N, 1});
    return res;
}

////////////////////////約数の個数を求める/////////////////////////////
////////////ここから
template <typename T>
map<T, T> prime_factor(T n) {
    map<T, T> ret;
    for (T i = 2; i * i <= n; i++) {
        T tmp = 0;
        while (n % i == 0) {
            tmp++;
            n /= i;
        }
        ret[i] = tmp;
    }
    if (n != 1) ret[n] = 1;
    return ret;
}
/*  divisor_num(n)
    入力：整数 n
    出力：nの約数の個数
    計算量：O(√n)
*/
template <typename T>
T divisor_num(T N) {
    map<T, T> pf = prime_factor(N);
    T ret = 1;
    for (auto p : pf) {
        ret *= (p.second + 1);
    }
    return ret;
}
//////////////////////////ここまで//////////////////////

////////////////////////配列の最大公約数//////////////////////
long long gcd_vec(vector<long long> const &A) {
    int size = (int)A.size();
    long long ret = A[0];
    for (int i = 1; i < size; i++) {
        ret = gcd(ret, A[i]);
    }
    return ret;
}
////////////配列の最小公倍数////////////////////////////////
long long lcm_vec(const vector<long long> &vec) {
    long long l = vec[0];
    for (int i = 0; i < vec.size() - 1; i++) {
        l = lcm(l, vec[i + 1]);
    }
    return l;
}
//////////////////////////////グリッド版dfs典型///////////////////////////////////////////////////


// 周囲 4 マスを探索するときに使う、差分を表す配列
int dx[4] = {1, 0, -1, 0};
int dy[4] = {0, 1, 0, -1};

// マス (x, y) がグリッド内のマスであるかを判定する
bool isvalid(int x, int y, int H, int W) {
    if(0 <= x && x < H && 0 <= y && y < W) {return true;}
    else {return false;}
}

// マス (x, y) の頂点番号
int getnum(int x, int y, int H, int W) {
    return (x * W + y);
}

// 頂点 v を始点とした深さ優先探索
void dfs(int v, vector<vector<int>> &G, vector<bool> &seen) {
    // 頂点 v を探索済みにする
    seen[v] = true;
    // G[v] に含まれる頂点 v2 について、
    for(auto v2 : G[v]) {
        // v2 がすでに探索済みならば、スキップする
        if(seen[v2]) {continue;}
        // v2 始点で深さ優先探索を行う (関数を再帰させる)
        dfs(v2, G, seen);
    }
    return;
}

int main() {
    // 入力を受け取る
    int H, W; cin >> H >> W;
    vector<vector<int>> grid(H, vector<int> (W, 0));    // grid[x][y]：マス (x, y) が白なら 0 、黒なら 1
    for(int x = 0; x < H; ++x) {
        string S; cin >> S;
        for(int y = 0; y < W; ++y) {
            if(S[y] == '#') {grid[x][y] = 1;}
            else if(S[y] == '.') {grid[x][y] = 0;}
        }
    }

    vector<vector<int>> G(H * W);   // グラフを表現する隣接リスト
    // グリッドの情報からグラフを作る
    for(int x = 0; x < H; ++x) {
        for(int y = 0; y < W; ++y) {
            // マス (x, y) が白マスなら、何もしない
            if(grid[x][y] == 0) {continue;}

            int v = getnum(x, y, H, W); // マス (x, y) に対応する頂点番号
            // マス (x, y) の上下左右のマスを探索
            for(int d = 0; d < 4; ++d) {
                int nx = x + dx[d], ny = y + dy[d];
                // マス (nx, ny) が盤内にあり、黒マスならば、対応する頂点同士を辺で結ぶ
                if(isvalid(nx, ny, H, W)) {
                    if(grid[nx][ny] == 0) {continue;}

                    int nv = getnum(nx, ny, H, W);  // マス (nx, ny) に対応する頂点番号
                    // 頂点 v から頂点 nv への辺を張る
                    G[v].push_back(nv);
                    // ダブルカウントされないよう、頂点 nv から頂点 v への辺は入れない
                }
            }
        }
    }

    vector<bool> seen(H * W, false);    // seen[v]：頂点 v が探索済みなら true, そうでないなら false
    //int ans = 0;    // 答えとなる変数
    for(int x = 0; x < H; ++x) {
        for(int y = 0; y < W; ++y) {
            // マス (x, y) が白マスなら、何もしない
            if(grid[x][y] == 0) {continue;}

            int v = getnum(x, y, H, W);
            // 頂点 v がすでに訪問済みであれば、スキップ
            if(seen[v]) {continue;}
            // そうでなければ、頂点 v を含む連結成分は未探索
            // 深さ優先探索で新たに訪問し、答えを 1 増やす
            dfs(v, G, seen);
            //ans++;
        }
    }
    // 答えを出力する
    //cout << ans << endl;

	return 0;
}
//********************************:コンビネーションnCk*************************:
ll nCk(ll a,ll b){
    ll p = 1;
    for(int i = 1; i <= b; i++){
        p *= a-i+1;
        p /= i;
    }
    return p;
}
//***********************************//////////////////////////////////////////////////

/**********************ダイクストラ*******************************/
/* dijkstra(G,s,dis,prev)
    入力：グラフ G, 開始点 s, 距離を格納する dis, 最短経路の前の点を記録するprev
    計算量：O(|E|log|V|)
    副作用：dis, prevが書き換えられる
*/

/*********************ここから**************/
struct Edge {
    long long to;
    long long cost;
};
using Graph = vector<vector<Edge>>;
using P = pair<long, int>;
const long long INF = 1LL << 60;
/*************ここまでは上に書く***********************:*/


void dijkstra(const Graph &G, int s, vector<long long> &dis, vector<int> &prev) {
    int N = G.size();
    dis.resize(N, INF);
    prev.resize(N, -1); // 初期化
    priority_queue<P, vector<P>, greater<P>> pq; 
    dis[s] = 0;
    pq.emplace(dis[s], s);
    while (!pq.empty()) {
        P p = pq.top();
        pq.pop();
        int v = p.second;
        if (dis[v] < p.first) {
            continue;
        }
        for (auto &e : G[v]) {
            if (dis[e.to] > dis[v] + e.cost) {
                dis[e.to] = dis[v] + e.cost;
                prev[e.to] = v; // 頂点 v を通って e.to にたどり着いた
                pq.emplace(dis[e.to], e.to);
            }
        }
    }
}
    /***************************************************************:*/
    /*************以下が経路復元の関数***************************/
    /* get_path(prev, t)
    入力：dijkstra で得た prev, ゴール t
    出力： t への最短路のパス
*/
vector<int> get_path(const vector<int> &prev, int t) {
    vector<int> path;
    for (int cur = t; cur != -1; cur = prev[cur]) {
        path.push_back(cur);
    }
    reverse(path.begin(), path.end()); // 逆順なのでひっくり返す
    return path;
}
/////**********************//////////////////////////////////////////////////

///巨大な数のmod////////////////////////////////////
ll string_mod(string s, ll mod){
    ll rest = 0;
    for(char c : s){
        ll v = c-'0';
        rest = (rest*10 + v) % mod;
    }
    return rest;
}
////////////////////////////////////////////////////

//////////////////////////////大きい数の演算//////////////////

vector<int> string_to_bigint(string S) {///////string から足し引きするためのvectorへ(逆順)//////
    int N = S.size(); // N = (文字列 S の長さ)
    vector<int> digit(N);
    for(int i = 0; i < N; ++i) {
        digit[i] = S[N - i - 1] - '0'; // 10^i の位の数
    }
    return digit;
}


string bigint_to_string(vector<int> digit) {/////////////////////vector からstring へ(逆順にすることで正しい順数字になっている)
    int N = digit.size(); // N = (配列 digit の長さ)
    string str = "";
    for(int i = N - 1; i >= 0; --i) {
        str += digit[i] + '0';
    }
    return str;
}

vector<int> carry_and_fix(vector<int> digit) {////////繰り上がり処理///////////////
    int N = digit.size();
    for(int i = 0; i < N - 1; ++i) {
        // 繰り上がり処理 (K は繰り上がりの回数)
        if(digit[i] >= 10) {
            int K = digit[i] / 10;
            digit[i] -= K * 10;
            digit[i + 1] += K;
        }
        // 繰り下がり処理 (K は繰り下がりの回数)
        if(digit[i] < 0) {
            int K = (-digit[i] - 1) / 10 + 1;
            digit[i] += K * 10;
            digit[i + 1] -= K;
        }
    }
    // 一番上の桁が 10 以上なら、桁数を増やすことを繰り返す
    while(digit.back() >= 10) {
        int K = digit.back() / 10;
        digit.back() -= K * 10;
        digit.push_back(K);
    }
    // 1 桁の「0」以外なら、一番上の桁の 0 (リーディング・ゼロ) を消す
    while(digit.size() >= 2 && digit.back() == 0) {
        digit.pop_back();
    }
    return digit;
}

vector<int> addition(vector<int> digit_a, vector<int> digit_b) {//////////////vector同士の足し算///////////
    int N = max(digit_a.size(), digit_b.size()); // a と b の大きい方
    vector<int> digit_ans(N); // 長さ N の配列 digit_ans を作る
    for(int i = 0; i < N; ++i) {
        digit_ans[i] = (i < digit_a.size() ? digit_a[i] : 0) + (i < digit_b.size() ? digit_b[i] : 0);
        // digit_ans[i] を digit_a[i] + digit_b[i] にする (範囲外の場合は 0)
    }
    return carry_and_fix(digit_ans); // 2-4 節「繰り上がり計算」の関数です
}

vector<int> subtraction(vector<int> digit_a, vector<int> digit_b) {//////////////////vetor同士の引き算//////////
    int N = max(digit_a.size(), digit_b.size()); // a と b の大きい方
    vector<int> digit_ans(N); // 長さ N の配列 digit_ans を作る
    for(int i = 0; i < N; ++i) {
        digit_ans[i] = (i < digit_a.size() ? digit_a[i] : 0) - (i < digit_b.size() ? digit_b[i] : 0);
        // digit_ans[i] を digit_a[i] - digit_b[i] にする (範囲外の場合は 0)
    }
    return carry_and_fix(digit_ans); // 2-4 節「繰り上がり計算」の関数です
}

vector<int> multiplication(vector<int> digit_a, vector<int> digit_b) {/////////////vector同士の掛け算////////
    int NA = digit_a.size(); // A の桁数
    int NB = digit_b.size(); // B の桁数
    vector<int> res(NA + NB - 1);
    for(int i = 0; i < NA; ++i) {
        for(int j = 0; j < NB; ++j) {
            res[i+j] += digit_a[i] * digit_b[j];
            // 答えの i+j の位に digit_a[i] * digit_b[j] を足す
        }
    }
    return carry_and_fix(res);
}

int compare_bigint(vector<int> digit_a, vector<int> digit_b) {
    int NA = digit_a.size(); // A の桁数
    int NB = digit_b.size(); // B の桁数
    if(NA > NB) return +1; // 左が大きい
    if(NA < NB) return -1; // 右が大きい
    for(int i = NA - 1; i >= 0; --i) {
        if(digit_a[i] > digit_b[i]) return +1; // 左が大きい
        if(digit_a[i] < digit_b[i]) return -1; // 右が大きい
    }
    return 0;
}

vector<int> division(vector<int> digit_a, vector<int> digit_b) {
    int NA = digit_a.size(), NB = digit_b.size();
    if(NA < NB) return { 0 };
    // ----- ステップ 1. A ÷ B の桁数を求める ----- //
    int D = NA - NB;
    // digit_a_partial : A の上 NB 桁を取り出したもの
    vector<int> digit_a_partial(digit_a.begin() + (NA - NB), digit_a.end());
    if(compare_bigint(digit_a_partial, digit_b) >= 0) {
        // (A の上 NB 桁) が B 以上だったら...？
        D = NA - NB + 1;
    }
    // ----- ステップ 2. A ÷ B を筆算で求める ----- //
    if(D == 0) return { 0 };
    vector<int> remain(digit_a.begin() + (D - 1), digit_a.end());
    vector<int> digit_ans(D);
    for(int i = D - 1; i >= 0; --i) {
        digit_ans[i] = 9;
        for(int j = 1; j <= 9; ++j) {
            vector<int> x = multiplication(digit_b, { j });
            if(compare_bigint(x, remain) == 1) {
                digit_ans[i] = j - 1;
                break;
            }
        }
        vector<int> x_result = multiplication(digit_b, { digit_ans[i] });
        remain = subtraction(remain, x_result);
        if(i >= 1) {
            // 新しく 10^(i-1) の位が降りてくる
            remain.insert(remain.begin(), digit_a[i - 1]);
        }
    }
    return digit_ans;
}

vector<int> vec_pow(ll a,ll b){
    string a_str = to_string(a);
    vector<int> ans = {1};
    vector<int> a_vec = string_to_bigint(a_str);
    rep(i,b){
        ans = multiplication(ans,a_vec);
    }
    return ans;
}

///////////////////////////////座標圧縮/////////////////////////////////////
template <typename T>
vector<T> compress(vector<T> &X) {
    // ソートした結果を vals に
    vector<T> vals = X;
    sort(vals.begin(), vals.end());
    // 隣り合う重複を削除(unique), 末端のゴミを削除(erase)
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    // 各要素ごとに二分探索で位置を求める
    for (int i = 0; i < (int)X.size(); i++) {
        X[i] = lower_bound(vals.begin(), vals.end(), X[i]) - vals.begin();
    }
    return vals;
}

////////////圧縮2

// ll n;

// vector<P> comp(n);
//     rep(i,n){
//         comp[i] = {a[i],i};
//         sort(all(comp));
//         vector<ll> order(n);
//         rep(i,n){
//             order[comp[i].second] = i;
//         }
//     }
/////////////////////////////////////////////////////////////////////////
///トポロジカルソート//////
vector<ll> topo_sort(vector<vector<ll>> &g,vector<bool> &seen,ll v,ll n){
    vector<ll> ans;
    auto dfs = [&](auto dfs,ll u) -> void{
        if(seen[u]) return;
        seen[u] = true;
        for(auto next_u : g[u]){
            if(seen[next_u]){
                continue;
            }
            else{
                dfs(dfs,next_u);
            }
        }
        ans.push_back(u);
    };
    rep(i,n){
        dfs(dfs,i);
    }
    reverse(all(ans));
    return ans;
}
//////////////////////////////////////////

//////////////平衡2分探索木/////////////////////////////

//操作まとめ
/*
- RBST<int> S; 　宣言
- S.insert(値);  挿入
- S.erase(値);  削除
- S.get(k-1);   k番目に小さい値を得る。k-1であることに注意大きいにしたければもともとをマイナスにする。
- S.sum()   和を求める
*/

template<class VAL> struct RBST {
    VAL SUM_UNITY = 0;                              // to be set

    unsigned int randInt() {
        static unsigned int tx = 123456789, ty = 362436069, tz = 521288629, tw = 88675123;
        unsigned int tt = (tx ^ (tx << 11));
        tx = ty; ty = tz; tz = tw;
        return (tw = (tw ^ (tw >> 19)) ^ (tt ^ (tt >> 8)));
    }

    struct NODE {
        NODE *left, *right;
        VAL val;                        // the value of the node
        int size;                       // the size of the subtree 
        VAL sum;                        // the value-sum of the subtree

        NODE() : val(SUM_UNITY), size(1), sum(SUM_UNITY) {
            left = right = NULL;
        }

        NODE(VAL v) : val(v), size(1), sum(v) {
            left = right = NULL;
        }

        /* additional update */
        inline void update() {

        }

        /* additional lazy-propagation */
        inline void push() {

            /* ex: reverse */
            /*
            if (this->rev) {
            swap(this->left, this->right);
            if (this->left) this->left->rev ^= true;
            if (this->right) this->right->rev ^= true;
            this->rev = false;
            }
            */
        }
    };


    ///////////////////////
    // root
    ///////////////////////

    NODE* root;
    RBST() : root(NULL) { }
    RBST(NODE* node) : root(node) { }


    ///////////////////////
    // basic operations
    ///////////////////////

    /* size */
    inline int size(NODE *node) {
        return !node ? 0 : node->size;
    }
    inline int size() {
        return this->size(this->root);
    }

    /* sum */
    inline VAL sum(NODE *node) {
        return !node ? SUM_UNITY : node->sum;
    }
    inline VAL sum() {
        return this->sum(this->root);
    }

    /* update, push */
    inline NODE* update(NODE *node) {
        node->size = size(node->left) + size(node->right) + 1;
        node->sum = sum(node->left) + sum(node->right) + node->val;
        node->update();
        return node;
    }

    inline void push(NODE *node) {
        if (!node) return;
        node->push();
    }

    /* lower_bound */
    inline int lowerBound(NODE *node, VAL val) {
        push(node);
        if (!node) return 0;
        if (val <= node->val) return lowerBound(node->left, val);
        else return size(node->left) + lowerBound(node->right, val) + 1;
    }
    inline int lowerBound(VAL val) {
        return this->lowerBound(this->root, val);
    }

    /* upper_bound */
    inline int upperBound(NODE *node, VAL val) {
        push(node);
        if (!node) return 0;
        if (val >= node->val) return size(node->left) + upperBound(node->right, val) + 1;
        else return upperBound(node->left, val);
    }
    inline int upperBound(VAL val) {
        return this->upperBound(this->root, val);
    }

    /* count */
    inline int count(VAL val) {
        return upperBound(val) - lowerBound(val);
    }

    /* get --- k: 0-index */
    inline VAL get(NODE *node, int k) {
        push(node);
        if (!node) return -1;
        if (k == size(node->left)) return node->val;
        if (k < size(node->left)) return get(node->left, k);
        else return get(node->right, k - size(node->left) - 1);
    }
    inline VAL get(int k) {
        return get(this->root, k);
    }


    ///////////////////////
    // merge-split
    ///////////////////////

    NODE* merge(NODE *left, NODE *right) {
        push(left);
        push(right);
        if (!left || !right) {
            if (left) return left;
            else return right;
        }
        if (randInt() % (left->size + right->size) < left->size) {
            left->right = merge(left->right, right);
            return update(left);
        }
        else {
            right->left = merge(left, right->left);
            return update(right);
        }
    }
    void merge(RBST add) {
        this->root = this->merge(this->root, add.root);
    }
    pair<NODE*, NODE*> split(NODE* node, int k) { // [0, k), [k, n)
        push(node);
        if (!node) return make_pair(node, node);
        if (k <= size(node->left)) {
            pair<NODE*, NODE*> sub = split(node->left, k);
            node->left = sub.second;
            return make_pair(sub.first, update(node));
        }
        else {
            pair<NODE*, NODE*> sub = split(node->right, k - size(node->left) - 1);
            node->right = sub.first;
            return make_pair(update(node), sub.second);
        }
    }
    RBST split(int k) {
        pair<NODE*, NODE*> sub = split(this->root, k);
        this->root = sub.first;
        return RBST(sub.second);
    }


    ///////////////////////
    // insert-erase
    ///////////////////////

    void insert(const VAL val) {
        pair<NODE*, NODE*> sub = this->split(this->root, this->lowerBound(val));
        this->root = this->merge(this->merge(sub.first, new NODE(val)), sub.second);
    }

    void erase(const VAL val) {
        if (!this->count(val)) return;
        pair<NODE*, NODE*> sub = this->split(this->root, this->lowerBound(val));
        this->root = this->merge(sub.first, this->split(sub.second, 1).second);
    }


    ///////////////////////
    // debug
    ///////////////////////

    void print(NODE *node) {
        if (!node) return;
        push(node);
        print(node->left);
        cout << node->val << " ";
        print(node->right);
    }
    void print() {
        cout << "{";
        print(this->root);
        cout << "}" << endl;
    }
};


//スパーステーブル///更新なしの区間の最小値
template< typename T >
struct SparseTable {
  vector< vector< T > > st;
  vector< int > lookup;

  SparseTable(const vector< T > &v) {
    int b = 0;
    while((1 << b) <= v.size()) ++b;
    st.assign(b, vector< T >(1 << b));
    for(int i = 0; i < v.size(); i++) {
      st[0][i] = v[i];
    }
    for(int i = 1; i < b; i++) {
      for(int j = 0; j + (1 << i) <= (1 << b); j++) {
        st[i][j] = min(st[i - 1][j], st[i - 1][j + (1 << (i - 1))]);
      }
    }
    lookup.resize(v.size() + 1);
    for(int i = 2; i < lookup.size(); i++) {
      lookup[i] = lookup[i >> 1] + 1;
    }
  }

  inline T rmq(int l, int r) {
    int b = lookup[r - l];
    return min(st[b][l], st[b][r - (1 << b)]);
  }
};

//例
int main() {
  int N, Q;
  scanf("%d", &N);
  vector< int > vs(N);
  for(int i = 0; i < N; i++) scanf("%d", &vs[i]);
  SparseTable< int > table(vs);//配列vsで初期化する
  scanf("%d", &Q);
  while(Q--) {
    int x, y;
    scanf("%d %d", &x, &y);
    printf("%d\n", table.rmq(x, y + 1));
  }
}

////////////部分列か////////////
//これは長さ1のとき
bool f(string &s,string &t){
    if(s.size() != t.size() + 1) return false;
    ll si = 0;
    rep(ti,t.size()){
        while(si < s.size() && s[si] != t[ti]) si++;
        if(si == s.size()) return false;
        si++;
    }
    return true;
}
/*-------------------------------------------------------------------
    ランレングス圧縮
-------------------------------------------------------------------*/
vector<pair<char, ll>> encode(const string& str) {
    ll n = (int)str.size();
    vector<pair<char, ll>> ret;
    for (int l = 0; l < n;) {
        int r = l + 1;
        for (; r < n && str[l] == str[r]; r++) {};
        ret.push_back({str[l], r - l});
        l = r;
    }
    return ret;
}
////////////////////////////////////////////////////////////