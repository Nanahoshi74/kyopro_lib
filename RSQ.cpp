#include <bits/stdc++.h>
//#include <atcoder/all>
//using namespace atcoder;
//using mint = modint998244353;
//using mint = modint1000000007;
// using mint;  /*このときmint::set_mod(mod)のようにしてmodを底にする*/
#define rep(i, n) for (int i = 0; i < (int)(n); i++)
#define repi(i,a,b) for(int i = a; i <= (int)(b); i++)
#define rng(i,a,b) for(int i = a; i < (int)(b); i++)
#define rrng(i,a,b) for(int i = a; i >= (int)(b); i--)
#define pb push_back
#define eb emplace_back
#define pob pop_back
#define si(a) (int)a.size()
#define all(a) a.begin(),a.end()
#define rall(a) a.rbegin(),a.rend()
#define ret(x) { cout<<(x)<<endl;}
typedef long long ll;
using namespace std;
using P = pair<ll,ll>;
const ll LINF = 1001002003004005006ll;
const int INF = 1001001001;
ll mul(ll a, ll b) { if (a == 0) return 0; if (LINF / a < b) return LINF; return min(LINF, a * b); }
ll mod(ll x, ll y){return (x % y + y) % y;}
char itoc(int x){return x + '0';}
int ctoi(char c){return c - '0';}

ll seg[1 << 20];

void add(ll ind,ll val){
    ind += 1 << 19;
    seg[ind] += val;
    while((ind /= 2) > 0){
        seg[ind] = seg[ind * 2] + seg[ind * 2 + 1];
    }
}

ll get_sum(ll ql,ll qr,ll sl = 1,ll sr = 1 << 19,ll pos = 1){
    //ちっとも被ってない
    if(qr <= sl || sr <= ql){
        return 0;
    }
    //セグメントの方がクエリに含まれている
    if(ql <= sl && sr <= qr){
        return seg[pos];
    }
    ll sm = (sl + sr)/2;
    ll l_val = get_sum(ql,qr,sl,sm,pos * 2);
    ll r_val = get_sum(ql,qr,sm,sr,pos * 2 + 1);
    return l_val + r_val;
}

int main(){

    ll n,q;
    cin >> n >> q;
    rep(i,q){
        ll com;
        cin >> com;
        if(com == 0){
            ll ind,val;
            cin >> ind >> val;
            add(ind,val);
        }
        else{
            ll l,r;
            cin >> l >> r;
            cout << get_sum(l,r+1) << endl;
        }
    }

    return 0;
}