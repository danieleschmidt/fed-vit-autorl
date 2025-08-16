"""Internationalization and localization support for Fed-ViT-AutoRL."""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path

from ..error_handling import with_error_handling, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for the system."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


@dataclass
class LocalizationConfig:
    """Configuration for localization."""
    default_language: SupportedLanguage = SupportedLanguage.ENGLISH
    fallback_language: SupportedLanguage = SupportedLanguage.ENGLISH
    auto_detect_locale: bool = True
    cache_translations: bool = True
    translation_dir: str = "translations"
    
    
class TranslationCache:
    """Thread-safe translation cache."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, str]] = {}
        self._access_count: Dict[str, int] = {}
        self._lock = threading.RLock()
    
    def get(self, language: str, key: str) -> Optional[str]:
        """Get translation from cache."""
        with self._lock:
            cache_key = f"{language}:{key}"
            if cache_key in self._cache:
                self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                return self._cache[cache_key]
            return None
    
    def put(self, language: str, key: str, translation: str) -> None:
        """Put translation in cache."""
        with self._lock:
            cache_key = f"{language}:{key}"
            
            # Evict if cache is full
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[cache_key] = translation
            self._access_count[cache_key] = 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_count:
            return
        
        lru_key = min(self._access_count.keys(), key=lambda k: self._access_count[k])
        del self._cache[lru_key]
        del self._access_count[lru_key]
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()


class I18nManager:
    """Internationalization manager for Fed-ViT-AutoRL."""
    
    def __init__(self, config: Optional[LocalizationConfig] = None):
        """Initialize i18n manager.
        
        Args:
            config: Localization configuration
        """
        self.config = config or LocalizationConfig()
        self.current_language = self.config.default_language
        
        # Translation storage
        self.translations: Dict[str, Dict[str, str]] = {}
        self.cache = TranslationCache() if self.config.cache_translations else None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load built-in translations
        self._load_builtin_translations()
        
        # Auto-detect locale if enabled
        if self.config.auto_detect_locale:
            self._auto_detect_language()
        
        logger.info(f"I18n manager initialized with language: {self.current_language.value}")
    
    def _load_builtin_translations(self) -> None:
        """Load built-in translations for core system messages."""
        
        # Core system messages in multiple languages
        builtin_translations = {
            SupportedLanguage.ENGLISH.value: {
                # Error messages
                "error.network.connection_failed": "Network connection failed",
                "error.model.training_failed": "Model training failed",
                "error.data.invalid_format": "Invalid data format",
                "error.security.unauthorized": "Unauthorized access",
                "error.validation.invalid_input": "Invalid input provided",
                
                # Status messages
                "status.training.started": "Training started",
                "status.training.completed": "Training completed successfully",
                "status.federation.round_started": "Federation round {round} started",
                "status.federation.round_completed": "Federation round {round} completed",
                "status.model.updated": "Model updated",
                "status.client.connected": "Client {client_id} connected",
                "status.client.disconnected": "Client {client_id} disconnected",
                
                # Configuration messages
                "config.privacy.enabled": "Privacy protection enabled",
                "config.compression.enabled": "Gradient compression enabled",
                "config.validation.strict": "Strict validation mode enabled",
                
                # Progress messages
                "progress.training.epoch": "Epoch {epoch}/{total_epochs}",
                "progress.federation.aggregating": "Aggregating updates from {count} clients",
                "progress.model.optimizing": "Optimizing model parameters",
                
                # Metrics
                "metrics.accuracy": "Accuracy",
                "metrics.loss": "Loss",
                "metrics.convergence_rate": "Convergence Rate",
                "metrics.communication_efficiency": "Communication Efficiency",
                "metrics.privacy_score": "Privacy Score",
            },
            
            SupportedLanguage.SPANISH.value: {
                # Error messages
                "error.network.connection_failed": "Falló la conexión de red",
                "error.model.training_failed": "Falló el entrenamiento del modelo",
                "error.data.invalid_format": "Formato de datos inválido",
                "error.security.unauthorized": "Acceso no autorizado",
                "error.validation.invalid_input": "Entrada inválida proporcionada",
                
                # Status messages
                "status.training.started": "Entrenamiento iniciado",
                "status.training.completed": "Entrenamiento completado exitosamente",
                "status.federation.round_started": "Ronda de federación {round} iniciada",
                "status.federation.round_completed": "Ronda de federación {round} completada",
                "status.model.updated": "Modelo actualizado",
                "status.client.connected": "Cliente {client_id} conectado",
                "status.client.disconnected": "Cliente {client_id} desconectado",
                
                # Configuration messages
                "config.privacy.enabled": "Protección de privacidad habilitada",
                "config.compression.enabled": "Compresión de gradientes habilitada",
                "config.validation.strict": "Modo de validación estricta habilitado",
                
                # Progress messages
                "progress.training.epoch": "Época {epoch}/{total_epochs}",
                "progress.federation.aggregating": "Agregando actualizaciones de {count} clientes",
                "progress.model.optimizing": "Optimizando parámetros del modelo",
                
                # Metrics
                "metrics.accuracy": "Precisión",
                "metrics.loss": "Pérdida",
                "metrics.convergence_rate": "Tasa de Convergencia",
                "metrics.communication_efficiency": "Eficiencia de Comunicación",
                "metrics.privacy_score": "Puntuación de Privacidad",
            },
            
            SupportedLanguage.FRENCH.value: {
                # Error messages
                "error.network.connection_failed": "Échec de la connexion réseau",
                "error.model.training_failed": "Échec de l'entraînement du modèle",
                "error.data.invalid_format": "Format de données invalide",
                "error.security.unauthorized": "Accès non autorisé",
                "error.validation.invalid_input": "Entrée invalide fournie",
                
                # Status messages
                "status.training.started": "Entraînement démarré",
                "status.training.completed": "Entraînement terminé avec succès",
                "status.federation.round_started": "Tour de fédération {round} démarré",
                "status.federation.round_completed": "Tour de fédération {round} terminé",
                "status.model.updated": "Modèle mis à jour",
                "status.client.connected": "Client {client_id} connecté",
                "status.client.disconnected": "Client {client_id} déconnecté",
                
                # Configuration messages
                "config.privacy.enabled": "Protection de la vie privée activée",
                "config.compression.enabled": "Compression des gradients activée",
                "config.validation.strict": "Mode de validation stricte activé",
                
                # Progress messages
                "progress.training.epoch": "Époque {epoch}/{total_epochs}",
                "progress.federation.aggregating": "Agrégation des mises à jour de {count} clients",
                "progress.model.optimizing": "Optimisation des paramètres du modèle",
                
                # Metrics
                "metrics.accuracy": "Précision",
                "metrics.loss": "Perte",
                "metrics.convergence_rate": "Taux de Convergence",
                "metrics.communication_efficiency": "Efficacité de Communication",
                "metrics.privacy_score": "Score de Confidentialité",
            },
            
            SupportedLanguage.GERMAN.value: {
                # Error messages
                "error.network.connection_failed": "Netzwerkverbindung fehlgeschlagen",
                "error.model.training_failed": "Modelltraining fehlgeschlagen",
                "error.data.invalid_format": "Ungültiges Datenformat",
                "error.security.unauthorized": "Unbefugter Zugriff",
                "error.validation.invalid_input": "Ungültige Eingabe bereitgestellt",
                
                # Status messages
                "status.training.started": "Training gestartet",
                "status.training.completed": "Training erfolgreich abgeschlossen",
                "status.federation.round_started": "Föderationsrunde {round} gestartet",
                "status.federation.round_completed": "Föderationsrunde {round} abgeschlossen",
                "status.model.updated": "Modell aktualisiert",
                "status.client.connected": "Client {client_id} verbunden",
                "status.client.disconnected": "Client {client_id} getrennt",
                
                # Configuration messages
                "config.privacy.enabled": "Datenschutz aktiviert",
                "config.compression.enabled": "Gradientenkompression aktiviert",
                "config.validation.strict": "Strenger Validierungsmodus aktiviert",
                
                # Progress messages
                "progress.training.epoch": "Epoche {epoch}/{total_epochs}",
                "progress.federation.aggregating": "Aggregierung von Updates von {count} Clients",
                "progress.model.optimizing": "Optimierung der Modellparameter",
                
                # Metrics
                "metrics.accuracy": "Genauigkeit",
                "metrics.loss": "Verlust",
                "metrics.convergence_rate": "Konvergenzrate",
                "metrics.communication_efficiency": "Kommunikationseffizienz",
                "metrics.privacy_score": "Datenschutz-Score",
            },
            
            SupportedLanguage.JAPANESE.value: {
                # Error messages
                "error.network.connection_failed": "ネットワーク接続に失敗しました",
                "error.model.training_failed": "モデルトレーニングに失敗しました",
                "error.data.invalid_format": "無効なデータ形式",
                "error.security.unauthorized": "不正なアクセス",
                "error.validation.invalid_input": "無効な入力が提供されました",
                
                # Status messages
                "status.training.started": "トレーニングが開始されました",
                "status.training.completed": "トレーニングが正常に完了しました",
                "status.federation.round_started": "フェデレーションラウンド{round}が開始されました",
                "status.federation.round_completed": "フェデレーションラウンド{round}が完了しました",
                "status.model.updated": "モデルが更新されました",
                "status.client.connected": "クライアント{client_id}が接続されました",
                "status.client.disconnected": "クライアント{client_id}が切断されました",
                
                # Configuration messages
                "config.privacy.enabled": "プライバシー保護が有効になりました",
                "config.compression.enabled": "勾配圧縮が有効になりました",
                "config.validation.strict": "厳密な検証モードが有効になりました",
                
                # Progress messages
                "progress.training.epoch": "エポック{epoch}/{total_epochs}",
                "progress.federation.aggregating": "{count}クライアントからの更新を集約中",
                "progress.model.optimizing": "モデルパラメータを最適化中",
                
                # Metrics
                "metrics.accuracy": "精度",
                "metrics.loss": "損失",
                "metrics.convergence_rate": "収束率",
                "metrics.communication_efficiency": "通信効率",
                "metrics.privacy_score": "プライバシースコア",
            },
            
            SupportedLanguage.CHINESE.value: {
                # Error messages
                "error.network.connection_failed": "网络连接失败",
                "error.model.training_failed": "模型训练失败",
                "error.data.invalid_format": "数据格式无效",
                "error.security.unauthorized": "未授权访问",
                "error.validation.invalid_input": "提供的输入无效",
                
                # Status messages
                "status.training.started": "训练已开始",
                "status.training.completed": "训练成功完成",
                "status.federation.round_started": "联邦轮次{round}已开始",
                "status.federation.round_completed": "联邦轮次{round}已完成",
                "status.model.updated": "模型已更新",
                "status.client.connected": "客户端{client_id}已连接",
                "status.client.disconnected": "客户端{client_id}已断开",
                
                # Configuration messages
                "config.privacy.enabled": "隐私保护已启用",
                "config.compression.enabled": "梯度压缩已启用",
                "config.validation.strict": "严格验证模式已启用",
                
                # Progress messages
                "progress.training.epoch": "周期{epoch}/{total_epochs}",
                "progress.federation.aggregating": "正在聚合来自{count}个客户端的更新",
                "progress.model.optimizing": "正在优化模型参数",
                
                # Metrics
                "metrics.accuracy": "准确率",
                "metrics.loss": "损失",
                "metrics.convergence_rate": "收敛率",
                "metrics.communication_efficiency": "通信效率",
                "metrics.privacy_score": "隐私分数",
            },
        }
        
        # Store translations
        with self._lock:
            for lang, translations in builtin_translations.items():
                if lang not in self.translations:
                    self.translations[lang] = {}
                self.translations[lang].update(translations)
    
    def _auto_detect_language(self) -> None:
        """Auto-detect user language from environment."""
        try:
            import locale
            
            # Get system locale
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                # Extract language code
                lang_code = system_locale.split('_')[0].lower()
                
                # Map to supported language
                for supported_lang in SupportedLanguage:
                    if supported_lang.value == lang_code:
                        self.current_language = supported_lang
                        logger.info(f"Auto-detected language: {lang_code}")
                        return
        
        except Exception as e:
            logger.warning(f"Failed to auto-detect language: {e}")
        
        # Fallback to default
        self.current_language = self.config.default_language
    
    @with_error_handling(max_retries=1, auto_recover=True)
    def set_language(self, language: Union[str, SupportedLanguage]) -> bool:
        """Set the current language.
        
        Args:
            language: Language to set
            
        Returns:
            True if successful
        """
        try:
            if isinstance(language, str):
                # Find matching language
                for lang in SupportedLanguage:
                    if lang.value == language.lower():
                        language = lang
                        break
                else:
                    logger.warning(f"Unsupported language: {language}")
                    return False
            
            with self._lock:
                self.current_language = language
                logger.info(f"Language set to: {language.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to set language: {e}")
            return False
    
    @with_error_handling(max_retries=1, auto_recover=True)
    def translate(
        self,
        key: str,
        language: Optional[Union[str, SupportedLanguage]] = None,
        **kwargs
    ) -> str:
        """Translate a message key to the specified language.
        
        Args:
            key: Translation key
            language: Target language (defaults to current language)
            **kwargs: Format parameters for the translation
            
        Returns:
            Translated message
        """
        # Determine target language
        if language is None:
            target_language = self.current_language
        elif isinstance(language, str):
            target_language = None
            for lang in SupportedLanguage:
                if lang.value == language.lower():
                    target_language = lang
                    break
            if target_language is None:
                target_language = self.current_language
        else:
            target_language = language
        
        lang_code = target_language.value
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(lang_code, key)
            if cached:
                try:
                    return cached.format(**kwargs) if kwargs else cached
                except KeyError:
                    # Format error, continue to regular translation
                    pass
        
        # Get translation
        with self._lock:
            translation = self._get_translation(key, lang_code)
        
        # Format with parameters
        try:
            formatted = translation.format(**kwargs) if kwargs else translation
        except KeyError as e:
            logger.warning(f"Missing format parameter for key '{key}': {e}")
            formatted = translation
        
        # Cache the result
        if self.cache:
            self.cache.put(lang_code, key, formatted)
        
        return formatted
    
    def _get_translation(self, key: str, language: str) -> str:
        """Get translation for a key in specified language."""
        # Try the requested language
        if language in self.translations and key in self.translations[language]:
            return self.translations[language][key]
        
        # Try fallback language
        fallback_lang = self.config.fallback_language.value
        if fallback_lang in self.translations and key in self.translations[fallback_lang]:
            logger.debug(f"Using fallback translation for key: {key}")
            return self.translations[fallback_lang][key]
        
        # Return key as last resort
        logger.warning(f"No translation found for key: {key}")
        return key
    
    def add_translations(
        self,
        language: Union[str, SupportedLanguage],
        translations: Dict[str, str]
    ) -> None:
        """Add custom translations for a language.
        
        Args:
            language: Target language
            translations: Dictionary of key-value translations
        """
        if isinstance(language, SupportedLanguage):
            lang_code = language.value
        else:
            lang_code = language.lower()
        
        with self._lock:
            if lang_code not in self.translations:
                self.translations[lang_code] = {}
            
            self.translations[lang_code].update(translations)
            
            # Clear cache for this language
            if self.cache:
                # Simple cache invalidation - could be more sophisticated
                self.cache.clear()
        
        logger.info(f"Added {len(translations)} translations for language: {lang_code}")
    
    def load_translations_from_file(self, filepath: str) -> bool:
        """Load translations from a JSON file.
        
        Args:
            filepath: Path to the translation file
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if not isinstance(data, dict):
                logger.error(f"Invalid translation file format: {filepath}")
                return False
            
            # Add translations for each language
            for lang_code, translations in data.items():
                if isinstance(translations, dict):
                    self.add_translations(lang_code, translations)
                else:
                    logger.warning(f"Invalid translations for language {lang_code}")
            
            logger.info(f"Loaded translations from: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load translations from {filepath}: {e}")
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return [lang.value for lang in SupportedLanguage]
    
    def get_current_language(self) -> str:
        """Get current language code."""
        return self.current_language.value
    
    def get_translation_coverage(self, language: Optional[str] = None) -> Dict[str, float]:
        """Get translation coverage statistics.
        
        Args:
            language: Language to check (defaults to all languages)
            
        Returns:
            Dictionary with coverage statistics
        """
        with self._lock:
            if language:
                languages = [language]
            else:
                languages = list(self.translations.keys())
            
            # Get reference key set from English (most complete)
            reference_keys = set(self.translations.get('en', {}).keys())
            
            coverage = {}
            for lang in languages:
                if lang in self.translations:
                    lang_keys = set(self.translations[lang].keys())
                    if reference_keys:
                        coverage[lang] = len(lang_keys & reference_keys) / len(reference_keys)
                    else:
                        coverage[lang] = 1.0
                else:
                    coverage[lang] = 0.0
            
            return coverage


# Global i18n manager instance
_global_i18n_manager: Optional[I18nManager] = None


def initialize_i18n(config: Optional[LocalizationConfig] = None) -> I18nManager:
    """Initialize global i18n manager."""
    global _global_i18n_manager
    _global_i18n_manager = I18nManager(config)
    return _global_i18n_manager


def get_i18n_manager() -> I18nManager:
    """Get global i18n manager, initializing if needed."""
    global _global_i18n_manager
    if _global_i18n_manager is None:
        _global_i18n_manager = I18nManager()
    return _global_i18n_manager


def t(key: str, language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for translation.
    
    Args:
        key: Translation key
        language: Target language
        **kwargs: Format parameters
        
    Returns:
        Translated message
    """
    return get_i18n_manager().translate(key, language, **kwargs)


def set_language(language: Union[str, SupportedLanguage]) -> bool:
    """Set global language.
    
    Args:
        language: Language to set
        
    Returns:
        True if successful
    """
    return get_i18n_manager().set_language(language)


def get_current_language() -> str:
    """Get current global language."""
    return get_i18n_manager().get_current_language()