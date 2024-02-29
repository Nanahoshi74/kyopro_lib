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

ll get(ll ind){
    ind += 1 << 19;
    ll ans = seg[ind];
    while((ind /= 2) > 0){
        ans += seg[ind];
    }
    return ans;
}

void add(ll ql,ll qr,ll val,ll sl = 1,ll sr = 1 << 19,ll pos = 1){
    //ちっとも被ってない
    if(qr <= sl || sr <= ql){
        return;
    }
    //セグメントの方がクエリに含まれている
    if(ql <= sl && sr <= qr){
        seg[pos] += val;
        return;
    }
    ll sm = (sl + sr)/2;
    add(ql,qr,val,sl,sm,pos * 2);
    add(ql,qr,val,sm,sr,pos * 2 + 1);
} 


int main(){

    ll n,q;
    cin >> n >> q;
    rep(i,q){
        ll com;
        cin >> com;
        if(com == 0){
            ll l,r,val;
            cin >> l >> r >> val;
            add(l,r + 1,val);
        }
        else{
            ll ind;
            cin >> ind;
            cout << get(ind) << endl;
        }
    }

    return 0;
}