package org.queueinc.hamlet.controller

data class Config(
    var iteration : Int,
    val dataset: String,
    val metric: String,
    val fairnessMetric: String,
    val sensitiveFeatures: String,
    val mode: String,
    val batchSize: Int,
    val timeBudget: Int,
    val seed: Int
)
