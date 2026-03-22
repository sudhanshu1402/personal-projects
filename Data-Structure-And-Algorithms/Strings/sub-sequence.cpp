
// subsequence is a sequencial char of any text !!

#include <bits/stdc++.h>
#include <string.h>

#include <iostream>
using namespace std;

// function for susequenece
bool isSubsequence(string& s1, string& s2) {
  // transverse for A
  int m = s1.length();
  int n = s2.length();
  int j = 0;
  // transverse  and always ++ i
  for (int i = 0; i < m && j < n; i++) {
    // if the condition matches, ++j & ++i
    if (s1[i] == s2[j]) {
      j++;
    }
  }
  // when to return true ?
  if (j == n) {
    return 1;
  } else {
    return 0;
  }
}

int main() {
  // take string 1
  string s1 = "ABCD";
  string s2 = "AD";

  cout << isSubsequence(s1, s2);
}
