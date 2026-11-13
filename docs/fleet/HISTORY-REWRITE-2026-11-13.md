# History rewrite, 2026-11-13

Both `cursor/toolchain-1.100` and `cursor/fleet-energy-harness-65da` were rewritten in one `git filter-branch --msg-filter` pass. The only change to a commit message is the removal of the line `Co-authored-by: David Charlot <dcharlot65-openie@users.noreply.github.com>` and any blank line that removal left at the end. Author, committer, dates, and trees were not changed. `main` was not rewritten.

## Counts

| Branch | Commits | Old tip | Rewritten tip |
| --- | --- | --- | --- |
| `cursor/toolchain-1.100` | 907 | `4abe63a78e8a6e6749bfea312b82d654981b44f9` | `796e63ab25a998fc4dda1f2f66e86b137aaf5cc8` |
| `cursor/fleet-energy-harness-65da` | 903 | `4d2c5378d1120996268929d0fc3fae3b2a0ca517` | `68ddf6765ca9f8a7ce4db6d7d23fcad66a54720e` |

910 unique commits are reachable from the two old tips. 877 of those kept their SHA. 33 were rewritten. 23 of the rewritten commits are on both branches, and each of those old SHAs maps to one new SHA. The rewritten tips above are the branch tips immediately after the filter, before this file was added.

## Tree proof

For every rewritten commit, `git rev-parse <old>^{tree}` equals `git rev-parse <new>^{tree}`. The author line and the committer line, including the timestamps, are identical. No rewritten message still contains the trailer. Result: 33 compared, 0 tree mismatches, 0 identity mismatches.

## Mapping

| Old SHA | New SHA | Tree | Branches | Subject |
| --- | --- | --- | --- | --- |
| `886fd46daf38eb021ee8634cb868855ab8c755a2` | `0257a74c58f965952007c65a81559b95db42d44b` | `3979f1891dd38f4b819ff8d7d10a0465bacf6115` | toolchain, fleet | Add acceptance receipts for six Ferric workstreams. |
| `831d7298733da0c0548ed21d3bb4299dce8e061b` | `2254e21dfcf71c4d6fae1172da260afa71f749e5` | `59042f6c169570ab967ec597aa8ed01e90e7b051` | toolchain, fleet | Add the Ferromotion SmolVLA action-chunk contract. |
| `fd3e7af6cecdd1c7afddeb68188303ebe452913c` | `dbbb426d24c9b4aee93d8d624a94cc6c7867dd54` | `f5b831d21cefa00de7943e8a59acea9d990b0729` | toolchain, fleet | Record Zeus and ML.ENERGY rules on track C energy receipts. |
| `3ee11c015e3478c7c4aa0409499346f79ab1c8ce` | `96e7fe5615d1ea6cca69cce2baef1ee2c1cdd2fc` | `afdeda1fb7bca5368616e843efe8ab0f33acc697` | toolchain, fleet | Freeze the SmolVLA chunk codec at version 2. |
| `c05228929efc3bcb4210b475dd959370fa69893a` | `99232a060ce26fedc2847d5028388e1c9725ae64` | `18eae08c57cd3e865b57f0be347fcc7fa0b5118d` | toolchain, fleet | Lock the observed v2 empty-horizon protocol vector. |
| `57f28268e86fcfbc9d4ee9290333616202bad5c2` | `9b60bfe1a0d4d015707d7a6a65c7752cc80e1282` | `0008222d7f619d53d42fc92845ca76ab74ccc78b` | toolchain, fleet | Add the fleet energy harness and pinned SmolVLA artifact list. |
| `03e591d1b87503e56075d46de71a66b7bfc7b07d` | `354a6438d67ed5223e9b3eb4ef3267bab6e2b330` | `21365d0871bd2897a18c1c6fdc8efa46a2c96d33` | toolchain, fleet | Run SmolVLA flow matching on the pinned weights. |
| `9db29b0382cbab200b36c424f9c4067376b76b2d` | `946a7896c23bb3e02976ae1e24660df86e07d01b` | `f01361697c40ea6039a8483f870752e7ecb4d189` | toolchain, fleet | Compare SmolVLA to an independent reference and dispatch it through wgpu. |
| `619ebb5321f94abf4713867d65be9cf9a91c511c` | `b94db81c6bb33cce3b3d19569028e5473fd3e38b` | `b84ba189d42446f587a655eff0daf4e500601e8c` | toolchain, fleet | Add the RADV session, f32 forward bounds, and the quiet energy driver. |
| `c8c700c2dbc9a1879ccdecc833892bfc18a0e71a` | `7a48e910d2523de9df381202777f6c3d87629a08` | `61d399b29cfe59759da474653ee97b9afd828882` | toolchain, fleet | Refuse a busy settle before the energy workload starts. |
| `0c04c6c7047d6120073757b32550ff634f7e8db6` | `d0089a4b4ba41bbdc8606a32a69861d5edbf4358` | `2fa3374cccaa881a74e101de00dd8c34e1890e8d` | toolchain, fleet | Record the LeRobot v0.7.0 preprocess delta and score kernel faults. |
| `167449ff518e20fc88a779525ae1176e9cdbd083` | `bf3c4a7a097ce9848ce43af77b1d7a66f54e3e97` | `4c64f186bb9eb62ca6fa09445df87a4a21b8c625` | toolchain, fleet | ci: dispatch the workflow and run it on cursor branches |
| `b99139406673800d8278693bc0bf6553a44938e5` | `83cb41fcf8925b05ff72c47b0bde9d039dacc8f0` | `a855dfd1753826f8e1ec214e02f58a4a67116774` | toolchain, fleet | Catch RoPE cross-term faults with a derived rotation invariant. |
| `6ba9215f24d1ab48e5c20d929abed790d8e28989` | `ed1f22ba0abd5e00ed69c7612cf08f1322e35e44` | `3ebaa81777adedd9ebb81678010170def6bc865d` | toolchain, fleet | Add a macmon meter beside RAPL for the quiet energy driver. |
| `4f06c6d648759f011f11b6676ab4c42ef7b482e4` | `32e4d0e7c9e7d6d2a2d9d11f4a51529da159776f` | `e9224b165764f0a169eea75637d2dd843f7d92ab` | toolchain, fleet | Align the quiet gate with ferrotherm c82c1e96. |
| `807d532e474a3dadce2fa85d7a52ff064a06def9` | `85131d1a4aa1afe14945cae6d07b543115269e74` | `e6499dd135a0ad37c41d53ba25d4cd1e7884f2f2` | toolchain, fleet | kv: widen cache scales with integer f16, not unpack2x16float |
| `bfac52e54d1fa1da1acd6a148ef809ac9c0ed6bc` | `7a6d8946a9d41be0a3a809d8c20bb6c2142a9eae` | `2b7b4d3c55115c86326bdfea122e55c9810487fa` | toolchain, fleet | ci: classify the packed IQ and NVFP4 formats, and decline a Metal pipeline the compiler rejects |
| `d990c06d796eefaadd1307041f13228efa5dc4ea` | `f6fc1047b087deb701607f8c9aa36e24ed139966` | `10e807a8023b43f3e268d27f4cd232ead253aeb7` | toolchain, fleet | smolvla: accept a libm ulp in the fault catalogue, not a real rmse change |
| `35bc79a8bda16a670c55913c4067f1ee40c9345a` | `a0ac46ddec9e7df6a154100d5a7eeefe311d0b40` | `3781131923e065f759c68b5b74e158d696536fec` | toolchain, fleet | Give each macmon field its own scope and refuse a zero DRAM total. |
| `83234fb1b06fc26d46be5e9314ab5371d440b96f` | `821427204b910dd916c0b8337f3d98ab84c29d70` | `595300bcebb6454bdd6ec95b92e159f7dbe9b062` | toolchain, fleet | qgemm: accumulate in a threadgroup tensor, not a cooperative tensor |
| `806cbe17075d848d1c83e4a9244d13ccf2873380` | `c0e598ece9e55a6712da50b3f2fc8ed02bca3fe1` | `e2c571e02d3d18a2f9e7a3fd8babb7b1f46234f2` | toolchain, fleet | Read macOS load1 from sysctl without unsafe code. |
| `298c6b6a60fbfc8c59d5a63d2b6466cbe995445f` | `041116e2e5c7896949de69086b49b207a41e688e` | `44253b816ba32c831dac38a92e6e9c547e5c4c51` | toolchain, fleet | ci: pin lavapipe to AVX2 so LLVM 21 does not SIGSEGV |
| `a999660dda508751961bddd86718b244f04dc0b6` | `6e76cddcb0745776245c31dfda1b87804533bfa3` | `161d91887e725064cf107f1897339eb0dcad4a06` | toolchain, fleet | smolvla: keep the fault catalogue bit-exact on Linux only |
| `afa210e96f89ca49f9d0f884d6f24ce381d89d82` | `60c15423492f08b7c0c3a5fcdac3f897c3be8bd7` | `16ef4f503d64f791f3a6eab7bdfa719e5d02dfa1` | fleet | ci: pin lavapipe with the Mesa token that actually drops AVX-512 |
| `bcf2a36105b083cea4a884a4c5f1d3607ac6186d` | `1fad2cc418559dc958e82ae7538f7abb948e2da4` | `628488f494d8d6ced68fad3814dd6be13fbc9b10` | toolchain | Pin the toolchain to Rust 1.100.0 and current GitHub Actions. |
| `55247d4c8726fcdce27feaaf8054714b5f1cff78` | `91a90fb8d94c8b853306cdc79247c9b68b7875dd` | `91314841f2e1c8521209598791c849a845eba1ab` | toolchain | Record cargo-deny policy and the upstream-blocked paste advisory. |
| `f4d5621fd401af5e1c713170eebb95a5c8fef893` | `3fc97219ed4de30baa97e6ef24f610c79285b94c` | `f2760c8b30368d44a12fa5b0395a4fee784d6608` | toolchain | Adapt call sites to the new crates and clear library clippy. |
| `88123e5f57f5422115f92ba39a4bd4868113125c` | `32c4031dea9adc0943e4cc8f4a5f9d3e95779941` | `85e33f1f9eb68e2ad8ae8ef490758a3d24ed6f2e` | toolchain | Take dependencies to their latest releases, including breaking majors. |
| `e4613150d7219cd7c1faf0dff01a2d8f3ccd35ab` | `9e80dba1c5e9de4d7084b92ce3c1eee5462a87bc` | `e485259d9f976b4fc717b8d15466f6a755acf12d` | toolchain | Clear example clippy without allows or deleted examples. |
| `9ba10c828091a50ae3dfd22b232c8dbc979bcbcf` | `5c1802e39289f7db04e886093993b9dc1aa3a40a` | `7d9ad7c9c33afa59d2ccb5753fdc1bf166b65916` | fleet | ci: run lavapipe tests one at a time |
| `249e8c7054beb7844fb99d9822983af076de5535` | `26ff0f84c43e68499f55c0ce621bfc6b64329f2a` | `4064fdd5728fbbb3952dc90a55ef8c2b4f7d950a` | toolchain | Restore the generation-tag rewrite and record the remeasured receipts. |
| `4abe63a78e8a6e6749bfea312b82d654981b44f9` | `796e63ab25a998fc4dda1f2f66e86b137aaf5cc8` | `8e61cde90fc0a437273743d66ed7d334b7aedd74` | toolchain | ci: pin lavapipe to the Mesa token that drops AVX-512 |
| `4d2c5378d1120996268929d0fc3fae3b2a0ca517` | `68ddf6765ca9f8a7ce4db6d7d23fcad66a54720e` | `84590e02a3afbe7108451d31bcef57b66d045eff` | fleet | smolvla: compare the fault catalogue within a libm ulp |
