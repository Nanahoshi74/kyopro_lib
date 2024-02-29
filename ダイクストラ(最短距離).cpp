#include <bits/stdc++.h>
#define rep(i, n) for (int i = 0; i < (int)(n); i++)
#define repi(i,a,b) for(int i = a; i <= (int)(b); i++)
#define rng(i,a,b) for(int i = a; i < (int)(b); i++)
#define rrng(i,a,b) for(int i = a; i >= (int)(b); i--)
#define pb push_back
#define em emplace_back
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


int main(){

    ll n,m;
    cin >> n >> m;
    vector<vector<P>> G(n);
    rep(i,m){
        ll a,b,cost;
        cin >> a >> b >> cost;
        a--,b--;
        G[a].emplace_back(b,cost);
        G[b].emplace_back(a,cost);
    }
    vector<ll> cur(n,LINF);
    vector<bool> seen(n,false);
    cur[0] = 0;
    priority_queue<P,vector<P>,greater<P>> pq;
    pq.emplace(cur[0],0);
    while(!pq.empty()){
        ll pos = pq.top().second; pq.pop();
        if(seen[pos]) continue;
        seen[pos] = true;
        rep(i,si(G[pos])){
            ll next = G[pos][i].first;
            ll cost = G[pos][i].second;
            if(cur[next] > cur[pos] + cost){
                cur[next] = cur[pos] + cost;
                pq.emplace(cur[next],next);
            }
        }
    }
    rep(i,n){
        if(cur[i] == LINF){
            cout << -1 << endl;
        }
        else{
            cout << cur[i] << endl;
        }
    }

    return 0;
}