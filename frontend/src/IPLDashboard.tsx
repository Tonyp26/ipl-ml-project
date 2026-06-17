import overviewData from "./data/overview.json"
import experimentData from "./data/experiment_results.json"
import backtestData from "./data/backtest_results.json"
import simulationData from "./data/simulation_results.json"

export default function IPLDashboard() {
  return (
    <div className="flex-1 p-8 bg-[#0D1117] text-white min-h-screen overflow-y-auto">
      {/* Header */}
      <header className="mb-8 border-b border-gray-800 pb-6">
        <h1 className="text-3xl font-bold text-white">
          {overviewData.projectOverview.title}
        </h1>
        <p className="text-gray-400 mt-1">
          {overviewData.projectOverview.tagline}
        </p>
      </header>

      {/* Research Notice */}
      <div className="mb-8 p-4 bg-yellow-900/30 border border-yellow-700 rounded-lg">
        <p className="text-yellow-200 text-sm font-semibold mb-1">Research Notice</p>
        <p className="text-yellow-200/80 text-sm">
          {overviewData.researchNotice}
        </p>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-5 gap-4 mb-10">
        {overviewData.projectOverview.stats.map((stat) => (
          <div key={stat.label} className="bg-[#161B22] rounded-lg p-4 text-center border border-gray-800">
            <div className="text-2xl font-bold text-blue-400">{stat.value}</div>
            <div className="text-sm text-gray-500 mt-1">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Description */}
      <div className="mb-10 bg-[#161B22] rounded-lg p-6 border border-gray-800">
        <p className="text-gray-300 leading-relaxed">
          {overviewData.projectOverview.description}
        </p>
      </div>

      {/* Section: Dataset Statistics */}
      <section className="mb-10">
        <h2 className="text-2xl font-bold mb-6 text-white">Dataset Statistics</h2>

        <div className="bg-[#161B22] rounded-lg p-6 border border-gray-800">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">Data Pipeline</h3>
          <div className="flex items-center gap-4 text-sm overflow-x-auto">
            <div className="bg-gray-800 rounded px-4 py-3 whitespace-nowrap">
              <span className="text-blue-400 font-mono">IPL.csv</span>
              <span className="text-gray-500 ml-2">283,678 rows</span>
            </div>
            <span className="text-gray-600 text-lg">→</span>
            <div className="bg-gray-800 rounded px-4 py-3 whitespace-nowrap">
              <span className="text-green-400 font-mono">matches.csv</span>
              <span className="text-gray-500 ml-2">1,193 matches</span>
            </div>
            <span className="text-gray-600 text-lg">→</span>
            <div className="bg-gray-800 rounded px-4 py-3 whitespace-nowrap">
              <span className="text-purple-400 font-mono">features</span>
              <span className="text-gray-500 ml-2">up to 48</span>
            </div>
            <span className="text-gray-600 text-lg">→</span>
            <div className="bg-gray-800 rounded px-4 py-3 whitespace-nowrap">
              <span className="text-orange-400 font-mono">model</span>
              <span className="text-gray-500 ml-2">RandomForest</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mt-4">
          <div className="bg-[#161B22] rounded-lg p-6 border border-gray-800">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Class Balance</h3>
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">Team 2 won</span>
                  <span className="text-white font-mono">55.2%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-4">
                  <div className="bg-blue-600 h-4 rounded-full" style={{ width: "55.2%" }}></div>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4 mt-3">
              <div className="flex-1">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">Team 1 won</span>
                  <span className="text-white font-mono">44.8%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-4">
                  <div className="bg-green-600 h-4 rounded-full" style={{ width: "44.8%" }}></div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#161B22] rounded-lg p-6 border border-gray-800">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Season Coverage</h3>
            <p className="text-gray-400 text-sm">
              19 seasons from <span className="text-white">2007/08</span> to <span className="text-white">2026</span>
            </p>
            <p className="text-gray-500 text-sm mt-2">
              10 current teams + 4 historical franchises
            </p>
            <p className="text-gray-500 text-sm mt-1">
              59 unique venues across India
            </p>
          </div>
        </div>
      </section>

      {/* Section: Feature Engineering Timeline */}
      <section className="mb-10">
        <h2 className="text-2xl font-bold mb-6 text-white">Feature Engineering Timeline</h2>

        <div className="bg-[#161B22] rounded-lg p-6 border border-gray-800">
          <div className="space-y-4">
            {experimentData.featureTimeline.map((phase, i) => (
              <div key={phase.phase} className="flex items-center gap-4">
                <div className="w-32 flex-shrink-0">
                  <span className={`font-mono text-sm font-bold ${
                    phase.phase === "Base" ? "text-gray-400" :
                    phase.phase === "+Elo" ? "text-green-400" :
                    phase.phase === "+Venue" ? "text-blue-400" :
                    "text-red-400"
                  }`}>
                    {phase.phase}
                  </span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <div className="w-16 text-right text-gray-500 text-sm">{phase.features}f</div>
                    <div className="flex-1 bg-gray-800 rounded-full h-3">
                      <div
                        className="h-3 rounded-full transition-all"
                        style={{
                          width: `${(phase.auc / 0.55) * 100}%`,
                          backgroundColor: phase.auc >= 0.45 ? "#3B82F6" : phase.auc >= 0.43 ? "#EAB308" : "#EF4444"
                        }}
                      ></div>
                    </div>
                    <div className="w-16 text-white font-mono text-sm">{phase.auc.toFixed(3)}</div>
                  </div>
                </div>
                <div className="w-48 text-gray-500 text-sm">{phase.note}</div>
                {i < experimentData.featureTimeline.length - 1 && (
                  <div className="absolute left-8 w-px h-4 bg-gray-700"></div>
                )}
              </div>
            ))}
          </div>
          <div className="mt-4 text-xs text-gray-600">Test AUC progression (0.50 = random baseline)</div>
        </div>
      </section>

      {/* Section: Model Comparison */}
      <section className="mb-10">
        <h2 className="text-2xl font-bold mb-6 text-white">Model Comparison</h2>

        <div className="bg-[#161B22] rounded-lg border border-gray-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-800/50 text-gray-400">
                <th className="text-left p-3">Model</th>
                <th className="text-right p-3">Features</th>
                <th className="text-right p-3">Accuracy</th>
                <th className="text-right p-3">Test AUC</th>
                <th className="text-right p-3">TS-CV AUC</th>
                <th className="text-center p-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {experimentData.models.map((model) => (
                <tr key={model.name} className={`border-t border-gray-800 ${
                  model.status === "production" ? "bg-green-900/20" :
                  model.status === "best-accuracy" ? "bg-blue-900/20" :
                  model.status === "noise" ? "bg-red-900/10" : ""
                }`}>
                  <td className="p-3 font-mono text-white">{model.name}</td>
                  <td className="p-3 text-right text-gray-400">{model.features}</td>
                  <td className={`p-3 text-right font-mono ${
                    model.accuracy >= 0.45 ? "text-green-400" : "text-red-400"
                  }`}>
                    {(model.accuracy * 100).toFixed(1)}%
                  </td>
                  <td className={`p-3 text-right font-mono ${
                    model.testAuc >= 0.45 ? "text-green-400" : "text-red-400"
                  }`}>
                    {model.testAuc.toFixed(3)}
                  </td>
                  <td className="p-3 text-right font-mono text-white">
                    {model.tsCvAuc.toFixed(3)} ± {model.tsCvStd.toFixed(3)}
                  </td>
                  <td className="p-3 text-center">
                    <span className={`text-xs px-2 py-1 rounded ${
                      model.status === "production" ? "bg-green-800 text-green-200" :
                      model.status === "best-accuracy" ? "bg-blue-800 text-blue-200" :
                      model.status === "baseline" ? "bg-gray-700 text-gray-300" :
                      "bg-red-900/50 text-red-300"
                    }`}>
                      {model.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Feature Importance */}
        <div className="bg-[#161B22] rounded-lg p-6 border border-gray-800 mt-4">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">Top Feature Importance</h3>
          <div className="space-y-2">
            {experimentData.featureImportance.slice(0, 12).map((f) => (
              <div key={f.feature} className="flex items-center gap-3">
                <div className="w-44 text-sm text-gray-300 font-mono truncate">{f.feature}</div>
                <div className="flex-1 bg-gray-800 rounded-full h-2.5">
                  <div
                    className="h-2.5 rounded-full bg-blue-500"
                    style={{ width: `${(f.importance / 0.076) * 100}%` }}
                  ></div>
                </div>
                <div className="w-12 text-right text-sm text-gray-500 font-mono">{f.importance.toFixed(3)}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Section: IPL 2025 Backtest Results */}
      <section className="mb-10">
        <h2 className="text-2xl font-bold mb-2 text-white">IPL 2025 Backtest Results</h2>
        <p className="text-gray-500 mb-6 text-sm">Out-of-sample evaluation: trained on ≤ 2024, tested on 2025</p>

        {/* Metric Cards */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          {[
            { label: "Accuracy", value: backtestData.metrics.accuracy, baseline: backtestData.baselines.accuracy, format: "pct" },
            { label: "ROC-AUC", value: backtestData.metrics.rocAuc, baseline: backtestData.baselines.rocAuc, format: "dec" },
            { label: "Brier Score", value: backtestData.metrics.brierScore, baseline: backtestData.baselines.brierScore, format: "dec" },
            { label: "Log Loss", value: backtestData.metrics.logLoss, baseline: backtestData.baselines.logLoss, format: "dec" },
          ].map((m) => (
            <div key={m.label} className="bg-[#161B22] rounded-lg p-4 border border-red-900/50">
              <div className="text-sm text-gray-500 mb-1">{m.label}</div>
              <div className="text-2xl font-bold text-red-400 font-mono">
                {m.format === "pct" ? `${(m.value * 100).toFixed(1)}%` : m.value.toFixed(3)}
              </div>
              <div className="text-xs text-gray-600 mt-1">baseline: {m.baseline.toFixed(3)}</div>
              <div className="text-xs text-red-500 mt-1">
                {m.label === "Brier Score" || m.label === "Log Loss"
                  ? "higher = worse"
                  : "below random"}
              </div>
            </div>
          ))}
        </div>

        {/* Standings Comparison */}
        <div className="bg-[#161B22] rounded-lg border border-gray-800 overflow-hidden mb-4">
          <div className="p-4 border-b border-gray-800">
            <h3 className="text-lg font-semibold text-gray-200">Predicted vs Actual Standings</h3>
            <div className="text-sm text-gray-500 mt-1">
              Top 4 overlap: <span className="text-red-400 font-bold">{backtestData.top4Overlap.overlap}/4</span> ({backtestData.top4Overlap.teams.join(", ")})
            </div>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-800/50 text-gray-400">
                <th className="text-left p-3">Pred Rank</th>
                <th className="text-left p-3">Team</th>
                <th className="text-right p-3">Pred Pts</th>
                <th className="text-right p-3">Actual Rank</th>
                <th className="text-right p-3">Actual Pts</th>
                <th className="text-center p-3">Match</th>
              </tr>
            </thead>
            <tbody>
              {backtestData.standings.map((s) => (
                <tr key={s.team} className="border-t border-gray-800">
                  <td className="p-3 text-gray-400">{s.predRank}</td>
                  <td className="p-3 text-white">{s.team}</td>
                  <td className="p-3 text-right font-mono text-gray-400">{s.predPts}</td>
                  <td className={`p-3 text-right font-mono ${s.match ? "text-green-400" : "text-red-400"}`}>
                    {s.actualRank}
                  </td>
                  <td className="p-3 text-right font-mono text-white">{s.actualPts}</td>
                  <td className="p-3 text-center">
                    {s.match ? (
                      <span className="text-green-400 text-xs">✓</span>
                    ) : (
                      <span className="text-red-400 text-xs">✗</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Most Confident Predictions */}
        <div className="bg-[#161B22] rounded-lg border border-gray-800 overflow-hidden">
          <div className="p-4 border-b border-gray-800">
            <h3 className="text-lg font-semibold text-gray-200">Top 20 Most Confident Predictions</h3>
            <p className="text-sm text-gray-500 mt-1">
              11 correct / 20 wrong (55%) — barely above coin flip
            </p>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-800/50 text-gray-400">
                <th className="p-3 text-right">#</th>
                <th className="p-3 text-left">Match</th>
                <th className="p-3 text-right">Confidence</th>
                <th className="p-3 text-center">Correct</th>
              </tr>
            </thead>
            <tbody>
              {backtestData.confidentPredictions.map((p) => (
                <tr key={p.rank} className="border-t border-gray-800">
                  <td className="p-3 text-right text-gray-500">{p.rank}</td>
                  <td className="p-3 text-gray-300">
                    <span className="text-gray-500">{p.date}</span>{" "}
                    {p.t1} vs {p.t2}
                  </td>
                  <td className="p-3 text-right font-mono text-white">
                    {Math.max(p.p1, 1 - p.p1).toFixed(1)}%
                  </td>
                  <td className="p-3 text-center">
                    {p.correct ? (
                      <span className="text-green-400">✓</span>
                    ) : (
                      <span className="text-red-400">✗</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Section: Tournament Simulation */}
      <section className="mb-10">
        <h2 className="text-2xl font-bold mb-2 text-white">Tournament Simulation Results</h2>
        <p className="text-gray-500 mb-6 text-sm">10,000 Monte Carlo iterations — IPL 2026 projected</p>

        {/* Probability Distribution */}
        <div className="bg-[#161B22] rounded-lg p-6 border border-gray-800 mb-4">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">Match Probability Distribution</h3>
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Mean:</span>{" "}
              <span className="text-white font-mono">{simulationData.matchProbabilities.mean.toFixed(3)}</span>
            </div>
            <div>
              <span className="text-gray-500">Range:</span>{" "}
              <span className="text-white font-mono">[{simulationData.matchProbabilities.min.toFixed(3)}, {simulationData.matchProbabilities.max.toFixed(3)}]</span>
            </div>
            <div>
              <span className="text-gray-500">Std:</span>{" "}
              <span className="text-white font-mono">{simulationData.matchProbabilities.std.toFixed(3)}</span>
            </div>
            <div className="text-yellow-400 text-xs flex items-center">
              Very narrow — model has little discrimination
            </div>
          </div>
        </div>

        {/* Points Table Projection */}
        <div className="bg-[#161B22] rounded-lg border border-gray-800 overflow-hidden mb-4">
          <div className="p-4 border-b border-gray-800">
            <h3 className="text-lg font-semibold text-gray-200">Projected Points Table</h3>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-800/50 text-gray-400">
                <th className="text-left p-3">Rank</th>
                <th className="text-left p-3">Team</th>
                <th className="text-right p-3">Mean</th>
                <th className="text-right p-3">P5</th>
                <th className="text-right p-3">P50</th>
                <th className="text-right p-3">P95</th>
                <th className="text-right p-3">Playoff%</th>
              </tr>
            </thead>
            <tbody>
              {simulationData.pointsTable.map((t) => (
                <tr key={t.team} className="border-t border-gray-800">
                  <td className="p-3 text-gray-500">{t.rank}</td>
                  <td className="p-3 text-white">{t.team}</td>
                  <td className="p-3 text-right font-mono text-white">{t.mean.toFixed(1)}</td>
                  <td className="p-3 text-right font-mono text-gray-500">{t.p5}</td>
                  <td className="p-3 text-right font-mono text-gray-400">{t.p50}</td>
                  <td className="p-3 text-right font-mono text-gray-500">{t.p95}</td>
                  <td className="p-3 text-right font-mono text-blue-400">{(t.playoffPct * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Playoff Odds */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-[#161B22] rounded-lg p-6 border border-gray-800">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Playoff Odds</h3>
            <div className="space-y-3">
              {simulationData.playoffOdds.map((t) => (
                <div key={t.team}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">{t.team}</span>
                    <span className="text-white font-mono">{(t.pTop4 * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div
                      className="h-2 rounded-full bg-blue-500"
                      style={{ width: `${t.pTop4 * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-[#161B22] rounded-lg p-6 border border-gray-800">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Tournament Winner Probability</h3>
            <div className="space-y-3">
              {simulationData.winnerOdds.map((t) => (
                <div key={t.team}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">{t.team}</span>
                    <span className="text-white font-mono">{(t.pWinner * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div
                      className="h-2 rounded-full bg-green-500"
                      style={{ width: `${t.pWinner * 100 * 6}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Simulation Parameters */}
        <div className="mt-4 bg-[#161B22] rounded-lg p-4 border border-gray-800">
          <h3 className="text-sm font-semibold text-gray-400 mb-2">Simulation Parameters</h3>
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div><span className="text-gray-500">Iterations:</span> <span className="text-white">{simulationData.simulationParams.iterations.toLocaleString()}</span></div>
            <div><span className="text-gray-500">Model:</span> <span className="text-white">{simulationData.simulationParams.model}</span></div>
            <div><span className="text-gray-500">Shrinkage:</span> <span className="text-white">{simulationData.simulationParams.shrinkage}</span></div>
            <div><span className="text-gray-500">Runtime:</span> <span className="text-white">{simulationData.simulationParams.runtime}</span></div>
          </div>
        </div>
      </section>

      {/* Section: Lessons Learned */}
      <section className="mb-10">
        <h2 className="text-2xl font-bold mb-6 text-white">Lessons Learned</h2>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-[#161B22] rounded-lg p-6 border border-green-900/50">
            <h3 className="text-lg font-semibold mb-4 text-green-400">What Worked</h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li>• Clean data pipeline from ball-by-ball to match-level</li>
              <li>• Chronological evaluation (no temporal leakage)</li>
              <li>• Rigorous feature audit (no future data leaked)</li>
              <li>• Honest reporting of below-random results</li>
              <li>• 2025 backtest proved out-of-sample failure</li>
            </ul>
          </div>

          <div className="bg-[#161B22] rounded-lg p-6 border border-red-900/50">
            <h3 className="text-lg font-semibold mb-4 text-red-400">What Didn't</h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li>• No feature set improved beyond random prediction</li>
              <li>• 48 features on 1,193 samples = too few data points</li>
              <li>• TS-CV AUC (~0.50) didn't translate to real 2025 performance</li>
              <li>• Momentum features were pure noise</li>
              <li>• Toss and venue effects are too weak to predict outcomes</li>
            </ul>
          </div>
        </div>

        <div className="bg-[#161B22] rounded-lg p-6 border border-blue-900/50 mt-4">
          <h3 className="text-lg font-semibold mb-4 text-blue-400">What Would Be Needed</h3>
          <ul className="space-y-2 text-sm text-gray-300">
            <li>• Player-level data (batting avg, strike rate, economy)</li>
            <li>• Real-time squad and injury information</li>
            <li>• Pitch conditions and weather data</li>
            <li>• Auction prices and team composition analysis</li>
            <li>• Many more seasons of data (IPL started in 2008 — only 19 seasons)</li>
          </ul>
        </div>

        {/* Honest Conclusion */}
        <div className="mt-6 bg-gray-800/50 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-3 text-white">The Honest Conclusion</h3>
          <p className="text-gray-300 italic leading-relaxed">
            "T20 cricket match outcomes are not predictable from pre-match features available in this dataset.
            The 20-over format has too much variance for historical team-level statistics to capture meaningful signal.
            This is not a failure of the model — it's a property of the sport."
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 pt-6 pb-8">
        <div className="flex justify-between items-center text-sm text-gray-600">
          <span>IPL ML Project — Educational purposes only</span>
          <a
            href="https://github.com/Tonyp26/ipl-ml-project"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:text-blue-300"
          >
            GitHub Repository →
          </a>
        </div>
      </footer>
    </div>
  )
}
