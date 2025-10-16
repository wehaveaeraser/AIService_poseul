package com.aiservice.poseul.ui.iot

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.aiservice.poseul.databinding.FragmentIotBinding

class IotFragment : Fragment() {

    private var _binding: FragmentIotBinding? = null
    private val binding get() = _binding!!
    private lateinit var iotViewModel: IotViewModel

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        iotViewModel = ViewModelProvider(this)[IotViewModel::class.java]
        _binding = FragmentIotBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        setupIotDevices()
        observeViewModel()
    }

    private fun setupIotDevices() {
        iotViewModel.iotDevices.observe(viewLifecycleOwner) { devices ->
            updateDeviceCards(devices)
        }
    }

    private fun updateDeviceCards(devices: List<IotDevice>) {
        // 에어컨 정보 업데이트
        val airConditioner = devices.find { it.type == "air_conditioner" }
        airConditioner?.let { device ->
            binding.airConditionerCard.visibility = View.VISIBLE
            binding.airConditionerStatus.text = if (device.isOnline) "온라인" else "오프라인"
            binding.airConditionerCurrentTemp.text = "${device.currentTemperature}°C"
            binding.airConditionerTargetTemp.text = "${device.targetTemperature}°C"
            binding.airConditionerStatus.setTextColor(
                if (device.isOnline) 
                    resources.getColor(com.aiservice.poseul.R.color.temperature_comfortable, null)
                else 
                    resources.getColor(com.aiservice.poseul.R.color.text_secondary, null)
            )
        } ?: run {
            binding.airConditionerCard.visibility = View.GONE
        }

        // 기타 IOT 기기들 업데이트
        val otherDevices = devices.filter { it.type != "air_conditioner" }
        if (otherDevices.isNotEmpty()) {
            binding.otherDevicesCard.visibility = View.VISIBLE
            updateOtherDevicesList(otherDevices)
        } else {
            binding.otherDevicesCard.visibility = View.GONE
        }
    }

    private fun updateOtherDevicesList(devices: List<IotDevice>) {
        // 실제 구현에서는 RecyclerView를 사용하는 것이 좋습니다
        val deviceInfo = devices.joinToString("\n") { device ->
            "${device.name}: ${if (device.isOnline) "온라인" else "오프라인"}"
        }
        binding.otherDevicesList.text = deviceInfo
    }

    private fun observeViewModel() {
        iotViewModel.isLoading.observe(viewLifecycleOwner) { isLoading ->
            binding.progressBar.visibility = if (isLoading) View.VISIBLE else View.GONE
        }

        iotViewModel.errorMessage.observe(viewLifecycleOwner) { error ->
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
