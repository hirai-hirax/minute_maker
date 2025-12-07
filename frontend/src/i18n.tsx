import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'

export type Language = 'ja' | 'en'

interface I18nContextValue {
  language: Language
  setLanguage: (lang: Language) => void
  toggleLanguage: () => void
}

const I18nContext = createContext<I18nContextValue>({
  language: 'ja',
  setLanguage: () => {},
  toggleLanguage: () => {},
})

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguage] = useState<Language>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('minute-maker-language') as Language | null
      if (stored === 'en' || stored === 'ja') return stored
    }
    return 'ja'
  })

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('minute-maker-language', language)
    }
  }, [language])

  const value = useMemo(
    () => ({
      language,
      setLanguage,
      toggleLanguage: () => setLanguage(prev => (prev === 'ja' ? 'en' : 'ja')),
    }),
    [language]
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  return useContext(I18nContext)
}
