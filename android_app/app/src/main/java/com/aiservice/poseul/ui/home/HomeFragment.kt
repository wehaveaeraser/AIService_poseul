package com.aiservice.poseul.ui.home

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.aiservice.poseul.databinding.FragmentHomeBinding
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import kotlinx.coroutines.*

class HomeFragment : Fragment() {

    private var _binding: FragmentHomeBinding? = null
    private val binding get() = _binding!!
    private lateinit var homeViewModel: HomeViewModel

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        homeViewModel = ViewModelProvider(this)[HomeViewModel::class.java]
        _binding = FragmentHomeBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        setupTemperatureDisplay()
        setupHeartRateChart()
        observeViewModel()
    }

    private fun setupTemperatureDisplay() {
        // 온도 예측 결과 표시
        homeViewModel.predictedTemperature.observe(viewLifecycleOwner) { temperature ->
            binding.temperatureValue.text = "${temperature}°C"
            updateTemperatureStatus(temperature)
        }
    }

    private fun updateTemperatureStatus(temperature: Float) {
        when {
            temperature < 20 -> {
                binding.temperatureStatus.text = "추움"
                binding.temperatureStatus.setTextColor(resources.getColor(com.aiservice.poseul.R.color.temperature_cold, null))
            }
            temperature > 28 -> {
                binding.temperatureStatus.text = "더움"
                binding.temperatureStatus.setTextColor(resources.getColor(com.aiservice.poseul.R.color.temperature_hot, null))
            }
            else -> {
                binding.temperatureStatus.text = "쾌적함"
                binding.temperatureStatus.setTextColor(resources.getColor(com.aiservice.poseul.R.color.temperature_comfortable, null))
            }
        }
    }

    private fun setupHeartRateChart() {
        // 심박수 차트 설정
        binding.heartRateChart.description.isEnabled = false
        binding.heartRateChart.setTouchEnabled(true)
        binding.heartRateChart.isDragEnabled = true
        binding.heartRateChart.setScaleEnabled(true)
        binding.heartRateChart.setPinchZoom(true)
        
        // 심박수 데이터 업데이트
        homeViewModel.heartRateData.observe(viewLifecycleOwner) { heartRates ->
            updateHeartRateChart(heartRates)
        }
    }

    private fun updateHeartRateChart(heartRates: List<Int>) {
        val entries = heartRates.mapIndexed { index, value ->
            Entry(index.toFloat(), value.toFloat())
        }

        val dataSet = LineDataSet(entries, "심박수").apply {
            color = resources.getColor(com.aiservice.poseul.R.color.heart_rate_red, null)
            setCircleColor(resources.getColor(com.aiservice.poseul.R.color.heart_rate_red, null))
            lineWidth = 2f
            circleRadius = 4f
            setDrawFilled(true)
            fillColor = resources.getColor(com.aiservice.poseul.R.color.heart_rate_red, null)
            fillAlpha = 30
        }

        val lineData = LineData(dataSet)
        binding.heartRateChart.data = lineData
        binding.heartRateChart.invalidate()
    }

    private fun observeViewModel() {
        homeViewModel.isLoading.observe(viewLifecycleOwner) { isLoading ->
            binding.progressBar.visibility = if (isLoading) View.VISIBLE else View.GONE
        }

        homeViewModel.errorMessage.observe(viewLifecycleOwner) { error ->
            if (error.isNotEmpty()) {
                binding.errorText.text = error
                binding.errorText.visibility = View.VISIBLE
            } else {
                binding.errorText.visibility = View.GONE
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
