package com.aiservice.poseul.service

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import java.net.HttpURLConnection
import java.net.URL
import java.io.OutputStreamWriter
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName

class ModelService {
    
    companion object {
        private const val SERVER_URL = "http://192.168.0.143:5000" // 실제 서버 IP 주소
        private const val PREDICT_ENDPOINT = "/predict"
        private const val HEALTH_ENDPOINT = "/health"
        private const val MODEL_INFO_ENDPOINT = "/model_info"
    }
    
    private val gson = Gson()
    
    data class PredictionRequest(
        @SerializedName("hr_mean") val hrMean: Double,
        @SerializedName("hrv_sdnn") val hrvSdnn: Double,
        @SerializedName("bmi") val bmi: Double,
        @SerializedName("mean_sa02") val meanSa02: Double,
        @SerializedName("gender") val gender: String
    )
    
    data class PredictionResponse(
        @SerializedName("success") val success: Boolean,
        @SerializedName("predicted_temperature") val predictedTemperature: Double,
        @SerializedName("temperature_category") val temperatureCategory: String,
        @SerializedName("input_data") val inputData: PredictionRequest?,
        @SerializedName("error") val error: String?
    )
    
    data class HealthResponse(
        @SerializedName("status") val status: String,
        @SerializedName("model_loaded") val modelLoaded: Boolean
    )
    
    suspend fun predictTemperature(
        heartRate: Int,
        hrvSdnn: Double,
        bmi: Double,
        meanSa02: Double,
        userGender: String
    ): PredictionResult = withContext(Dispatchers.IO) {
        try {
            // 서버 상태 확인
            if (!checkServerHealth()) {
                return@withContext PredictionResult.Error("서버에 연결할 수 없습니다.")
            }
            
            // 예측 요청 데이터 준비
            val request = PredictionRequest(
                hrMean = heartRate.toDouble(),
                hrvSdnn = hrvSdnn,
                bmi = bmi,
                meanSa02 = meanSa02,
                gender = if (userGender.lowercase() == "female") "F" else "M"
            )
            
            // HTTP 요청 수행
            val response = makePredictionRequest(request)
            
            if (response.success) {
                PredictionResult.Success(
                    temperature = response.predictedTemperature.toFloat(),
                    category = response.temperatureCategory
                )
            } else {
                PredictionResult.Error(response.error ?: "예측 실패")
            }
            
        } catch (e: Exception) {
            PredictionResult.Error("예측 중 오류 발생: ${e.message}")
        }
    }
    
    private suspend fun checkServerHealth(): Boolean = withContext(Dispatchers.IO) {
        try {
            val url = URL("$SERVER_URL$HEALTH_ENDPOINT")
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 5000
            connection.readTimeout = 5000
            
            val responseCode = connection.responseCode
            if (responseCode == HttpURLConnection.HTTP_OK) {
                val response = connection.inputStream.bufferedReader().use { it.readText() }
                val healthResponse = gson.fromJson(response, HealthResponse::class.java)
                healthResponse.modelLoaded
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }
    
    private suspend fun makePredictionRequest(request: PredictionRequest): PredictionResponse = withContext(Dispatchers.IO) {
        val url = URL("$SERVER_URL$PREDICT_ENDPOINT")
        val connection = url.openConnection() as HttpURLConnection
        
        connection.requestMethod = "POST"
        connection.setRequestProperty("Content-Type", "application/json")
        connection.doOutput = true
        connection.connectTimeout = 10000
        connection.readTimeout = 10000
        
        // 요청 데이터 전송
        val requestJson = gson.toJson(request)
        val outputStream = connection.outputStream
        val writer = OutputStreamWriter(outputStream)
        writer.write(requestJson)
        writer.flush()
        writer.close()
        
        // 응답 읽기
        val responseCode = connection.responseCode
        val responseText = if (responseCode == HttpURLConnection.HTTP_OK) {
            connection.inputStream.bufferedReader().use { it.readText() }
        } else {
            connection.errorStream.bufferedReader().use { it.readText() }
        }
        
        gson.fromJson(responseText, PredictionResponse::class.java)
    }
    
    suspend fun getModelInfo(): String = withContext(Dispatchers.IO) {
        try {
            val url = URL("$SERVER_URL$MODEL_INFO_ENDPOINT")
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 5000
            connection.readTimeout = 5000
            
            val responseCode = connection.responseCode
            if (responseCode == HttpURLConnection.HTTP_OK) {
                connection.inputStream.bufferedReader().use { it.readText() }
            } else {
                "모델 정보를 가져올 수 없습니다."
            }
        } catch (e: Exception) {
            "모델 정보 조회 실패: ${e.message}"
        }
    }
    
    suspend fun loadModel(): Boolean = withContext(Dispatchers.IO) {
        // 서버 기반 모델이므로 서버 상태만 확인
        checkServerHealth()
    }
    
    suspend fun retrainModel(newData: List<TrainingData>): Boolean = withContext(Dispatchers.IO) {
        try {
            // 실제 구현에서는 새로운 데이터로 모델을 재학습합니다
            // 1. 데이터 전처리
            // 2. 모델 학습
            // 3. 모델 검증
            // 4. 모델 저장
            
            // 시뮬레이션을 위한 대기
            delay(5000)
            true
        } catch (e: Exception) {
            false
        }
    }
}

sealed class PredictionResult {
    data class Success(
        val temperature: Float,
        val category: String
    ) : PredictionResult()
    
    data class Error(
        val message: String
    ) : PredictionResult()
}

data class TrainingData(
    val heartRate: Int,
    val roomTemperature: Float,
    val humidity: Float,
    val userAge: Int,
    val userGender: String,
    val actualTemperature: Float
)
