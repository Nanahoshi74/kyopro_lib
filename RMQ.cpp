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

ll init_value = (1LL << 31)-1;
ll seg[1 << 20];

void set_value(ll pos,ll val){
    pos += 1 << 19;
    seg[pos] = val;
    while((pos /= 2) > 0){
        seg[pos] = min(seg[pos * 2],seg[pos * 2 + 1]);
    }
}

ll get_min(ll ql,ll qr,ll sl = 0, ll sr = 1 << 19,ll pos = 1){
    //ちっとも被ってない
    if(qr <= sl || sr <= ql) return init_value;
    //完全に包まれる
    if(ql <= sl && sr <= qr) return seg[pos];
    ll sm = (sl + sr)/2;
    ll lmin = get_min(ql,qr,sl,sm,pos * 2);
    ll rmin = get_min(ql,qr,sm,sr,pos * 2 + 1);
    return min(lmin,rmin);
}

int main(){

    ll n,q;
    cin >> n >> q;
    rep(i,n){
        set_value(i,init_value);
    }
    rep(i,q){
        ll com,x,y;
        cin >> com >> x >> y;
        if(com == 0){
            set_value(x,y);
        }
        else{
            cout << get_min(x,y+1) << endl;
        }
    }

    return 0;
}