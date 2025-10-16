package com.aiservice.poseul.ui.user

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.aiservice.poseul.databinding.FragmentUserBinding
import com.google.android.material.dialog.MaterialAlertDialogBuilder

class UserFragment : Fragment() {

    private var _binding: FragmentUserBinding? = null
    private val binding get() = _binding!!
    private lateinit var userViewModel: UserViewModel

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        userViewModel = ViewModelProvider(this)[UserViewModel::class.java]
        _binding = FragmentUserBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        setupUserInfo()
        setupButtons()
        observeViewModel()
    }

    private fun setupUserInfo() {
        userViewModel.userInfo.observe(viewLifecycleOwner) { userInfo ->
            if (userInfo != null) {
                binding.userIdText.text = userInfo.userId
                binding.userNameText.text = userInfo.name
                binding.userEmailText.text = userInfo.email
                binding.loginButton.visibility = View.GONE
                binding.logoutButton.visibility = View.VISIBLE
                binding.userInfoCard.visibility = View.VISIBLE
            } else {
                binding.loginButton.visibility = View.VISIBLE
                binding.logoutButton.visibility = View.GONE
                binding.userInfoCard.visibility = View.GONE
            }
        }
    }

    private fun setupButtons() {
        binding.loginButton.setOnClickListener {
            showLoginDialog()
        }

        binding.logoutButton.setOnClickListener {
            showLogoutDialog()
        }

        binding.retrainButton.setOnClickListener {
            showRetrainDialog()
        }
    }

    private fun showLoginDialog() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("로그인")
            .setMessage("로그인 기능은 추후 구현될 예정입니다.")
            .setPositiveButton("확인") { dialog, _ ->
                dialog.dismiss()
            }
            .show()
    }

    private fun showLogoutDialog() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("로그아웃")
            .setMessage("정말 로그아웃하시겠습니까?")
            .setPositiveButton("로그아웃") { _, _ ->
                userViewModel.logout()
            }
            .setNegativeButton("취소") { dialog, _ ->
                dialog.dismiss()
            }
            .show()
    }

    private fun showRetrainDialog() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("모델 재학습")
            .setMessage("모델을 재학습하시겠습니까? 이 과정은 시간이 걸릴 수 있습니다.")
            .setPositiveButton("재학습 시작") { _, _ ->
                userViewModel.startRetraining()
            }
            .setNegativeButton("취소") { dialog, _ ->
                dialog.dismiss()
            }
            .show()
    }

    private fun observeViewModel() {
        userViewModel.isRetraining.observe(viewLifecycleOwner) { isRetraining ->
            binding.retrainProgress.visibility = if (isRetraining) View.VISIBLE else View.GONE
            binding.retrainButton.isEnabled = !isRetraining
            binding.retrainButton.text = if (isRetraining) "재학습 중..." else "모델 재학습"
        }

        userViewModel.retrainMessage.observe(viewLifecycleOwner) { message ->
            if (message.isNotEmpty()) {
                binding.retrainStatus.text = message
                binding.retrainStatus.visibility = View.VISIBLE
            } else {
                binding.retrainStatus.visibility = View.GONE
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
