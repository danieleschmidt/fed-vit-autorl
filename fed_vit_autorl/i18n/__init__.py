"""Internationalization and localization support for Fed-ViT-AutoRL."""

from typing import Dict, Optional, Any
import json
import os
import logging

logger = logging.getLogger(__name__)

class I18nManager:
    """Global internationalization manager."""

    def __init__(self):
        self.current_locale = 'en'
        self.translations: Dict[str, Dict[str, str]] = {}
        self.fallback_locale = 'en'
        self._load_translations()

    def _load_translations(self):
        """Load translation files."""
        translations_dir = os.path.join(os.path.dirname(__file__), 'translations')

        # Default English translations
        self.translations['en'] = {
            'federated_learning': 'Federated Learning',
            'autonomous_vehicle': 'Autonomous Vehicle',
            'vision_transformer': 'Vision Transformer',
            'privacy_preserved': 'Privacy Preserved',
            'training_complete': 'Training Complete',
            'model_updated': 'Model Updated',
            'error_occurred': 'Error Occurred',
            'performance_optimized': 'Performance Optimized',
            'security_validated': 'Security Validated',
            'client_connected': 'Client Connected',
            'aggregation_started': 'Aggregation Started',
            'round_completed': 'Round Completed',
            'inference_ready': 'Inference Ready'
        }

        # Spanish translations
        self.translations['es'] = {
            'federated_learning': 'Aprendizaje Federado',
            'autonomous_vehicle': 'Vehículo Autónomo',
            'vision_transformer': 'Transformador de Visión',
            'privacy_preserved': 'Privacidad Preservada',
            'training_complete': 'Entrenamiento Completo',
            'model_updated': 'Modelo Actualizado',
            'error_occurred': 'Error Ocurrido',
            'performance_optimized': 'Rendimiento Optimizado',
            'security_validated': 'Seguridad Validada',
            'client_connected': 'Cliente Conectado',
            'aggregation_started': 'Agregación Iniciada',
            'round_completed': 'Ronda Completada',
            'inference_ready': 'Inferencia Lista'
        }

        # French translations
        self.translations['fr'] = {
            'federated_learning': 'Apprentissage Fédéré',
            'autonomous_vehicle': 'Véhicule Autonome',
            'vision_transformer': 'Transformateur de Vision',
            'privacy_preserved': 'Confidentialité Préservée',
            'training_complete': 'Formation Terminée',
            'model_updated': 'Modèle Mis à Jour',
            'error_occurred': 'Erreur Survenue',
            'performance_optimized': 'Performance Optimisée',
            'security_validated': 'Sécurité Validée',
            'client_connected': 'Client Connecté',
            'aggregation_started': 'Agrégation Démarrée',
            'round_completed': 'Tour Terminé',
            'inference_ready': 'Inférence Prête'
        }

        # German translations
        self.translations['de'] = {
            'federated_learning': 'Föderiertes Lernen',
            'autonomous_vehicle': 'Autonomes Fahrzeug',
            'vision_transformer': 'Vision Transformer',
            'privacy_preserved': 'Datenschutz Gewährleistet',
            'training_complete': 'Training Abgeschlossen',
            'model_updated': 'Modell Aktualisiert',
            'error_occurred': 'Fehler Aufgetreten',
            'performance_optimized': 'Leistung Optimiert',
            'security_validated': 'Sicherheit Validiert',
            'client_connected': 'Client Verbunden',
            'aggregation_started': 'Aggregation Gestartet',
            'round_completed': 'Runde Abgeschlossen',
            'inference_ready': 'Inferenz Bereit'
        }

        # Japanese translations
        self.translations['ja'] = {
            'federated_learning': '連合学習',
            'autonomous_vehicle': '自動運転車',
            'vision_transformer': 'ビジョントランスフォーマー',
            'privacy_preserved': 'プライバシー保護',
            'training_complete': '訓練完了',
            'model_updated': 'モデル更新',
            'error_occurred': 'エラー発生',
            'performance_optimized': 'パフォーマンス最適化',
            'security_validated': 'セキュリティ検証',
            'client_connected': 'クライアント接続',
            'aggregation_started': '集約開始',
            'round_completed': 'ラウンド完了',
            'inference_ready': '推論準備完了'
        }

        # Chinese translations
        self.translations['zh'] = {
            'federated_learning': '联邦学习',
            'autonomous_vehicle': '自动驾驶汽车',
            'vision_transformer': '视觉变换器',
            'privacy_preserved': '隐私保护',
            'training_complete': '训练完成',
            'model_updated': '模型更新',
            'error_occurred': '发生错误',
            'performance_optimized': '性能优化',
            'security_validated': '安全验证',
            'client_connected': '客户端连接',
            'aggregation_started': '聚合开始',
            'round_completed': '轮次完成',
            'inference_ready': '推理就绪'
        }

        logger.info(f"Loaded translations for {len(self.translations)} locales")

    def set_locale(self, locale: str) -> bool:
        """Set current locale."""
        if locale in self.translations:
            self.current_locale = locale
            logger.info(f"Locale set to: {locale}")
            return True
        else:
            logger.warning(f"Locale '{locale}' not supported, using fallback")
            return False

    def t(self, key: str, locale: Optional[str] = None) -> str:
        """Translate a key to current or specified locale."""
        target_locale = locale or self.current_locale

        if target_locale in self.translations:
            translations = self.translations[target_locale]
            if key in translations:
                return translations[key]

        # Fallback to default locale
        if self.fallback_locale in self.translations:
            fallback_translations = self.translations[self.fallback_locale]
            if key in fallback_translations:
                return fallback_translations[key]

        # Return key if no translation found
        return key

    def get_supported_locales(self) -> list:
        """Get list of supported locales."""
        return list(self.translations.keys())

# Global i18n manager instance
i18n = I18nManager()

def translate(key: str, locale: Optional[str] = None) -> str:
    """Global translation function."""
    return i18n.t(key, locale)

def set_global_locale(locale: str) -> bool:
    """Set global locale."""
    return i18n.set_locale(locale)
